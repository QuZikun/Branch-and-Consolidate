"""Reverse KL on teacher top-k tokens plus one aggregate tail probability.

Let S be the teacher top-k set. The student is normalized over the full TP
vocabulary. Compute KL([p_S, sum(p_not_S)] || [q_S, sum(q_not_S)]).
Teacher probabilities are floored and renormalized to handle zero tail mass
and quantization overshoot. This is a coarsened KL, not full-vocabulary KL.
"""
import torch

from opd.common import require

REVERSE_MODE = 'reverse_kl_topk_tail'


def token_negative_feedback_alpha(alpha, shape, device):
    """Validate alpha and align scalar/per-row/per-token input to token axes."""
    alpha = torch.as_tensor(alpha, device=device)
    require(alpha.is_floating_point(), 'Negative-feedback alpha must be floating point')
    require(not alpha.requires_grad, 'Negative-feedback alpha is a schedule, not a learned tensor')
    require(bool(torch.isfinite(alpha).all()) and bool(((alpha >= 0) & (alpha <= 1)).all()),
            'Negative-feedback alpha must be finite and in [0, 1]')
    shape = tuple(shape)
    if alpha.ndim == 0:
        return alpha.expand(shape)
    if tuple(alpha.shape) == shape:
        return alpha
    if len(shape) == 2 and tuple(alpha.shape) in ((shape[0],), (shape[0], 1)):
        return alpha.reshape(shape[0], 1).expand(shape)
    require(False, 'Negative-feedback alpha shape differs from logits token axes')


def teacher_partition(log_probs, probability_floor):
    """Return a proper K+1 distribution, including BF16 top-k mass > 1 cases."""
    head = log_probs.exp()
    raw_mass = head.sum(-1)
    tail = (1.0 - raw_mass).clamp_min(0.0)
    distribution = torch.cat((head, tail.unsqueeze(-1)), dim=-1).clamp_min(probability_floor)
    distribution = distribution / distribution.sum(-1, keepdim=True)
    return distribution, raw_mass


class VocabParallelReverseKL(torch.autograd.Function):
    """Analytic gradient through the full-vocabulary student normalization."""

    @staticmethod
    def forward(ctx, logits, teacher_logps, teacher_ids, clamp, probability_floor,
                negative_feedback_alpha=None):
        from megatron.core.parallel_state import (
            get_tensor_model_parallel_group, get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        require(clamp is None, 'reverse_kl_topk_tail uses probability_floor; set log_prob_min_clamp=null')
        group = get_tensor_model_parallel_group()
        rank, world = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        shape, width = logits.shape[:-1], logits.shape[-1]
        alpha = None
        if negative_feedback_alpha is not None:
            alpha = token_negative_feedback_alpha(negative_feedback_alpha, shape, logits.device).reshape(-1).to(dtype)
        local = logits.reshape(-1, width).to(dtype)
        # kai-verl's teacher manager emits int32 IDs; gather/scatter require int64.
        ids = teacher_ids.reshape(local.shape[0], -1).long()
        k = ids.shape[-1]
        require(k < width * world, 'Top-k must leave a tail bucket')
        start, end = rank * width, (rank + 1) * width
        owned = (ids >= start) & (ids < end)
        valid = (ids >= 0) & (ids < width * world)
        local_ids = (ids - start).masked_fill(~owned, 0)

        def reduce(value, op=torch.distributed.ReduceOp.SUM):
            if world > 1:
                torch.distributed.all_reduce(value, op=op, group=group)
            return value

        maximum = reduce(local.amax(-1), torch.distributed.ReduceOp.MAX)
        shifted = local - maximum.unsqueeze(-1)
        exponentials = shifted.exp()
        normalizer = reduce(exponentials.sum(-1))
        log_probs = shifted - normalizer.log().unsqueeze(-1)
        probs = exponentials / normalizer.unsqueeze(-1)
        local_head = probs.gather(-1, local_ids) * owned
        head_log_probs = reduce(log_probs.gather(-1, local_ids) * owned)
        head = head_log_probs.exp() * valid

        # Sum the actual non-top-k probabilities. Avoid 1 - sum(topk), which
        # loses tiny student tails to float cancellation at confident tokens.
        tail_probs = probs.clone()
        tail_probs.scatter_add_(-1, local_ids, -local_head)
        tail = reduce(tail_probs.sum(-1)).clamp_min(0.0)
        tail_log_probs = tail.clamp_min(torch.finfo(dtype).tiny).log()
        teacher, teacher_mass = teacher_partition(
            teacher_logps.reshape(local.shape[0], k).to(dtype), probability_floor)
        teacher_logs = teacher.log()
        head_ratio = head_log_probs - teacher_logs[:, :k]
        tail_ratio = tail_log_probs - teacher_logs[:, k]
        head_terms = head * head_ratio
        loss = head_terms.sum(-1) + tail * tail_ratio

        # For bucket b(i), dL/dz_i = p_i * (a_b(i) - sum_b p_b*a_b),
        # where a_b=log(p_b/q_b)+1. All probabilities include the TP-wide Z.
        head_derivative = head_ratio + 1
        tail_derivative = tail_ratio + 1
        if alpha is not None and bool((alpha != 0).any()):
            # Raw advantage A_b=log(q_b/p_b): retain A>=0, multiply A<0
            # by 1-alpha. Its potential is the piecewise weighted divergence
            # sum_b w_b * [p_b log(p_b/q_b)-p_b+q_b], with w=1 for p<=q
            # and 1-alpha for p>q. dL/dp_b=w_b*log(p_b/q_b).
            # Recenter this transformed derivative below through full softmax.
            head_weight = torch.where(head_ratio > 0, 1 - alpha.unsqueeze(-1), 1.)
            tail_weight = torch.where(tail_ratio > 0, 1 - alpha, 1.)
            transformed_loss = (head_weight * (head_terms - head + teacher[:, :k])).sum(-1)
            transformed_loss += tail_weight * (tail * tail_ratio - tail + teacher[:, k])
            unchanged = alpha == 0
            # Preserve exact existing values/gradients for self and all alpha=0
            # rows, including those sharing a microbatch with external routes.
            loss = torch.where(unchanged, loss, transformed_loss)
            head_derivative = torch.where(unchanged.unsqueeze(-1), head_derivative, head_weight * head_ratio)
            tail_derivative = torch.where(unchanged, tail_derivative, tail_weight * tail_ratio)
            head_terms = head_terms * head_weight
        expected = (head * head_derivative).sum(-1) + tail * tail_derivative
        base = tail_derivative - expected
        correction = local_head * (head_derivative - tail_derivative.unsqueeze(-1))
        ctx.save_for_backward(probs, local_ids, base, correction)
        ctx.input_shape, ctx.input_dtype = logits.shape, logits.dtype

        # Retain the existing top-k overlap diagnostics, with reverse-KL head
        # contributions for overlap_token_advantage (tail has no individual id).
        local_scores, local_top_ids = torch.topk(log_probs, min(k, width), dim=-1)
        local_top_ids = local_top_ids + start
        if world > 1:
            scores = [torch.empty_like(local_scores) for _ in range(world)]
            candidates = [torch.empty_like(local_top_ids) for _ in range(world)]
            torch.distributed.all_gather(scores, local_scores, group=group)
            torch.distributed.all_gather(candidates, local_top_ids, group=group)
            positions = torch.cat(scores, -1).topk(k, dim=-1).indices
            student_ids = torch.cat(candidates, -1).gather(-1, positions)
        else:
            student_ids = local_top_ids
        overlap = (ids.unsqueeze(-1) == student_ids.unsqueeze(-2)).any(-1) & valid
        overlap_count = overlap.sum(-1)
        advantage = -(head_terms * overlap).sum(-1) / overlap_count.clamp_min(1)
        outputs = (loss.reshape(shape), head.sum(-1).reshape(shape), teacher_mass.reshape(shape),
                   overlap_count.reshape(shape), advantage.reshape(shape))
        ctx.mark_non_differentiable(*outputs[1:])
        return outputs

    @staticmethod
    def backward(ctx, grad_loss, *_diagnostics):
        probs, ids, base, correction = ctx.saved_tensors
        gradient = probs * base.unsqueeze(-1)
        gradient.scatter_add_(-1, ids, correction)
        gradient *= grad_loss.reshape(-1, 1)
        outputs = (gradient.reshape(ctx.input_shape).to(ctx.input_dtype), None, None, None, None, None)
        return outputs[:len(ctx.needs_input_grad)]


def register_reverse_kl():
    """Register before any native DistillationConfig dataclass is constructed."""
    from verl.trainer.distillation import losses
    if REVERSE_MODE not in losses.DISTILLATION_LOSS_REGISTRY:
        # The native adapter only aggregates these per-token outputs/mass metrics;
        # the logits kernel installed on the student supplies the reverse math.
        losses.register_distillation_loss(losses.DistillationLossSettings(names=REVERSE_MODE, use_topk=True))(
            losses.compute_forward_kl_topk)


def make_reverse_distillation_config(**kwargs):
    """Hydra target: register only when actor code constructs its loss config.

    Ray's worker setup hook runs before accelerator assignment. Avoid importing
    this module/verl there; their imports may initialize CUDA with stale visibility.
    """
    register_reverse_kl()
    from verl.workers.config import DistillationLossConfig
    require(kwargs.get('loss_mode') == REVERSE_MODE, 'Reverse config factory received another loss mode')
    return DistillationLossConfig(**kwargs)


def install_reverse_kl(chunk_tokens, probability_floor):
    from opd.native import chunked_kl_apply
    import verl.trainer.distillation.megatron.losses as native
    register_reverse_kl()
    signature = (REVERSE_MODE, chunk_tokens, probability_floor)
    old = getattr(native._VocabParallelKLDivergence, '_opd_signature', None)
    if old:
        require(old == signature, 'Cannot change the loss in an initialized worker')
        return
    require(not getattr(native._VocabParallelKLDivergence, '_opd_chunked', False),
            'A forward-KL kernel is already installed in this worker')

    def apply(logits, teacher_logps, teacher_ids, clamp, negative_feedback_alpha=None):
        return VocabParallelReverseKL.apply(logits, teacher_logps, teacher_ids, clamp, probability_floor,
                                           negative_feedback_alpha)

    class ChunkedReverseKL:
        _opd_signature = signature

        @staticmethod
        def apply(logits, teacher_logps, teacher_ids, clamp, negative_feedback_alpha=None):
            return chunked_kl_apply(apply, logits, teacher_logps, teacher_ids, clamp, chunk_tokens,
                                    negative_feedback_alpha=negative_feedback_alpha)

    native._VocabParallelKLDivergence = ChunkedReverseKL
    print(f'[opd] reverse KL: teacher top-k + tail; {chunk_tokens}-token chunks; '
          f'teacher probability floor={probability_floor}; full-vocab student normalization', flush=True)
