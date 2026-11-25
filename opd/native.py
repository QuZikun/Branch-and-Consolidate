"""kai-verl training adapters; forward math stays native, reverse math lives in losses.py."""
from contextlib import contextmanager
import os
import math
import time
import socket
from pathlib import Path
from opd.common import SOURCES, require, fingerprint, write_json

def logging_env(environ):
    """Send credentials/network settings only to the logging actor via RPC."""
    keys = ('WANDB_API_KEY', 'WANDB_BASE_URL', 'WANDB_ENTITY', 'WANDB_INIT_TIMEOUT',
            'WANDB_HTTP_PROXY', 'WANDB_HTTPS_PROXY', 'WANDB_RUN_GROUP',
            'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
            'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
            'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'SSL_CERT_FILE', 'SSL_CERT_DIR')
    result = {key: environ[key] for key in keys if key in environ}
    # Ray workers bypass proxies for internal services. The logging actor must
    # explicitly replace BOTH cases, including when the driver omitted a key;
    # otherwise its inherited NO_PROXY=* can silently disable W&B's proxy.
    # Normalize casing because Python and the W&B Go service differ in priority.
    for key in ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
        value = environ.get(key) or environ.get(key.upper()) or ''
        if key == 'no_proxy' and not value:
            value = 'localhost,127.0.0.1,::1'
        result[key] = result[key.upper()] = value
    return result


def prepare_tracking(config):
    """Connect W&B before loading models; fit() reuses this exact tracker."""
    from omegaconf import OmegaConf
    from verl.utils.tracking import Tracking
    import wandb
    run_dir = Path(config.trainer.default_local_dir).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    timeout = float(os.environ.get('WANDB_INIT_TIMEOUT') or '180')
    require(math.isfinite(timeout) and timeout > 0, 'WANDB_INIT_TIMEOUT must be positive')
    require(os.environ.get('WANDB_API_KEY'), 'Set WANDB_API_KEY before starting')
    os.environ['WANDB_INIT_TIMEOUT'] = str(timeout)
    os.environ['WANDB_DIR'] = str(run_dir)
    os.environ['WANDB_MODE'] = 'online'
    # A failed pre-model handshake may be retried in the same directory.
    # Native trainer independently resumes optimizer/RNG/dataloader checkpoints.
    os.environ['WANDB_RUN_ID'] = 'opd-'  + fingerprint(str(run_dir))[:20]
    os.environ['WANDB_RESUME'] = 'allow'
    report = dict(stage='initializing', hostname=socket.gethostname(), timeout_seconds=timeout,
                  run_id=os.environ['WANDB_RUN_ID'], model_initialization_started=False,
                  proxy_configured=any(os.environ.get(k) for k in (
                      'HTTPS_PROXY', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'WANDB_HTTPS_PROXY')),
                  proxy_bypass_all=any('*' in os.environ.get(k, '').split(',')
                                       for k in ('NO_PROXY', 'no_proxy')))
    write_json(run_dir / 'wandb_init.json', report)
    print('[opd] Connecting W&B before model loading; timeout=%ss' % timeout, flush=True)
    print('[opd] W&B network: proxy_configured=%s; proxy_bypass_all=%s' % (
        report['proxy_configured'], report['proxy_bypass_all']), flush=True)
    tracker = None
    try:
        tracker = Tracking(project_name=config.trainer.project_name,
                           experiment_name=config.trainer.experiment_name,
                           default_backend=list(config.trainer.logger),
                           config=OmegaConf.to_container(config, resolve=True))
        require(wandb.run is not None, 'W&B did not create a run')
        report.update(stage='ready', url=wandb.run.url)
        write_json(run_dir / 'wandb_init.json', report)
        print('[opd] W&B ready: ' + str(wandb.run.url) + '; starting model initialization.', flush=True)
        return tracker
    except Exception as exc:
        report.update(stage='failed', error_type=type(exc).__name__)
        write_json(run_dir / 'wandb_init.json', report)
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise RuntimeError('W&B initialization failed BEFORE model loading. See ' + str(run_dir / 'wandb_init.json')
                           + ' and ' + str(run_dir / 'wandb/latest-run/logs/debug-internal.log')) from exc


@contextmanager
def reuse_tracking(tracker):
    # Native fit imports Tracking locally. Scope this adapter to fit() in the
    # single logging actor so it cannot create a second W&B run/handshake.
    import verl.utils.tracking as tracking_module
    original = tracking_module.Tracking
    tracking_module.Tracking = lambda *args, **kwargs: tracker
    try:
        yield
    finally:
        tracking_module.Tracking = original


def chunked_kl_apply(native_apply, logits, teacher_logps, teacher_ids, clamp, chunk_tokens=1024,
                     negative_feedback_alpha=None):
    """Checkpoint each token chunk's intermediates without changing its KL kernel.

    Only the token axes are chunked. Every TP rank still normalizes over the
    complete distributed vocabulary. No prompt/response offset is changed.
    Disabling early-stop keeps all TP collectives in the original recompute order.
    Snapshot logits before checkpoint because native sampled-token cross entropy
    may consume the original FP32 logits in place after this call. CloneBackward
    preserves the model gradient; the checkpoint input must never be detached.
    """
    import torch
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    require(chunk_tokens > 0, 'KL chunk size must be positive')
    require(logits.ndim == 3 and teacher_logps.shape == teacher_ids.shape
            and logits.shape[:2] == teacher_ids.shape[:2], 'KL shape mismatch')
    shape = logits.shape[:2]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_logps = teacher_logps.reshape(-1, teacher_logps.shape[-1])
    flat_ids = teacher_ids.reshape(-1, teacher_ids.shape[-1])
    flat_alpha = None
    if negative_feedback_alpha is not None:
        from opd.losses import token_negative_feedback_alpha
        flat_alpha = token_negative_feedback_alpha(negative_feedback_alpha, shape, logits.device).reshape(-1)
    chunks = []
    for offset in range(0, flat_logits.shape[0], chunk_tokens):
        inputs = (flat_logits[offset:offset + chunk_tokens].unsqueeze(0),
                  flat_logps[offset:offset + chunk_tokens].unsqueeze(0),
                  flat_ids[offset:offset + chunk_tokens].unsqueeze(0), clamp)
        if flat_alpha is not None:
            inputs = (*inputs, flat_alpha[offset:offset + chunk_tokens].unsqueeze(0))
        if torch.is_grad_enabled() and logits.requires_grad:
            # Clone outside checkpoint, so replay cannot see later mutations.
            inputs = (inputs[0].clone(), *inputs[1:])
            with set_checkpoint_early_stop(False):
                outputs = checkpoint(native_apply, *inputs, use_reentrant=False, preserve_rng_state=False)
        else:
            outputs = native_apply(*inputs)
        require(len(outputs) == 5, 'Native KL output contract changed')
        chunks.append(outputs)
    require(bool(chunks), 'Empty KL logits')
    return tuple(torch.cat([part[i].reshape(-1) for part in chunks]).reshape(shape) for i in range(5))


def install_chunked_kl(chunk_tokens=1024):
    import verl.trainer.distillation.megatron.losses as native
    if getattr(native._VocabParallelKLDivergence, '_opd_chunked', False):
        require(native._VocabParallelKLDivergence._opd_chunk_tokens == chunk_tokens,
                'Cannot change KL chunk size in an initialized worker')
        return
    original_apply = native._VocabParallelKLDivergence.apply

    class ChunkedKL:
        _opd_chunked = True
        _opd_chunk_tokens = chunk_tokens

        @staticmethod
        def apply(logits, teacher_logps, teacher_ids, clamp):
            return chunked_kl_apply(original_apply, logits, teacher_logps, teacher_ids, clamp, chunk_tokens)

    native._VocabParallelKLDivergence = ChunkedKL
    print(f'[opd] native top-k KL: {chunk_tokens}-token chunks, checkpointed backward, full-vocab normalization', flush=True)


def negative_feedback_topk_loss(student_logits, data, distillation_config, data_format, config):
    """Expand per-query alpha using the same native THD/BSHD token packing.

    ``opd_negative_alpha`` is a floating tensor [microbatch_queries]. The native
    padding adapter leaves this row field intact while making teacher tensors
    jagged. Repeat each alpha over its teacher sequence, then apply the identical
    CP split/alignment to teacher probabilities, IDs, and alpha. No response
    offsets or tool masks are inferred from sequence lengths here.
    """
    import torch
    from verl.models.mcore.util import preprocess_thd_engine, preprocess_bshd_engine
    import verl.trainer.distillation.megatron.losses as native
    require(config.strategy == 'megatron', 'Negative feedback requires the Megatron top-k adapter')
    require(distillation_config.distillation_loss.loss_mode == 'reverse_kl_topk_tail',
            'Negative feedback is defined only for reverse_kl_topk_tail')
    teacher_logps, teacher_ids = data['teacher_logprobs'], data['teacher_ids']
    require(teacher_logps.is_nested and teacher_ids.is_nested,
            'Negative feedback expects native jagged teacher tensors')
    row_alpha = data['opd_negative_alpha']
    require(row_alpha.ndim == 1 and len(row_alpha) == teacher_logps.shape[0],
            'Negative-feedback alpha rows lost during microbatch splitting')
    offsets = teacher_logps.offsets()
    require(torch.equal(offsets, teacher_ids.offsets()), 'Teacher offsets differ')
    row_alpha = row_alpha.to(device=teacher_logps.device, dtype=torch.float32)
    values = torch.repeat_interleave(row_alpha, offsets.diff()).unsqueeze(-1)
    alpha = torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
    require(data_format in ('thd', 'bshd'), 'Unknown native data format: ' + data_format)
    preprocess = preprocess_thd_engine if data_format == 'thd' else preprocess_bshd_engine
    teacher_logps, *_ = preprocess(teacher_logps, pre_process=True)
    teacher_ids, *_ = preprocess(teacher_ids, pre_process=True)
    alpha, *_ = preprocess(alpha, pre_process=True)
    alpha = alpha.squeeze(-1)
    require(student_logits.shape[:2] == teacher_logps.shape[:2] == teacher_ids.shape[:2] == alpha.shape,
            'Negative-feedback alpha packing differs from logits/teacher rows')
    outputs = native._VocabParallelKLDivergence.apply(
        student_logits, teacher_logps, teacher_ids,
        distillation_config.distillation_loss.log_prob_min_clamp, alpha)
    names = ('distillation_losses', 'student_mass', 'teacher_mass', 'overlap_count', 'overlap_token_advantage')
    require(len(outputs) == len(names), 'Native KL output contract changed')
    return dict(zip(names, outputs))


def monitored_distillation_loss(*args, **kwargs):
    """Native GKD loss unchanged; append detached sums over response tokens only."""
    import torch
    from verl.trainer.distillation.losses import distillation_ppo_loss
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding
    if kwargs.get('student_logits') is not None and 'opd_negative_alpha' in kwargs['data'].keys():
        return negative_feedback_topk_loss(
            kwargs['student_logits'], kwargs['data'], kwargs['distillation_config'],
            kwargs.get('data_format', 'thd'), kwargs['config'])
    result = distillation_ppo_loss(*args, **kwargs)
    if kwargs.get('student_logits') is not None:
        return result
    output, data = kwargs['model_output'], kwargs['data']
    loss, metrics = result
    if 'opd_route_weight' in data.keys():
        from opd.balanced import weighted_route_loss
        routed_values = no_padding_2_padding(output['distillation_losses'], data)
        route_mask = data['response_mask']
        route_mask = route_mask.to_padded_tensor(False) if route_mask.is_nested else route_mask
        loss = weighted_route_loss(routed_values, route_mask, data['opd_route_weight'],
                                   int(data['dp_size']), kwargs['distillation_config'].distillation_loss.loss_max_clamp)
        metrics['distillation/loss'] = Metric(AggregationType.SUM, loss.detach().item())
    with torch.no_grad():
        values = no_padding_2_padding(output['distillation_losses'], data).detach().float().clamp_min(0.0)
        max_loss = kwargs['distillation_config'].distillation_loss.loss_max_clamp
        if max_loss is not None:
            values = values.clamp_max(max_loss)
        teacher_mass = no_padding_2_padding(output['teacher_mass'], data).detach().float()
        student_mass = (no_padding_2_padding(output['student_mass'], data).detach().float()
                        if output.get('student_mass') is not None else None)
        overlap_count = (no_padding_2_padding(output['overlap_count'], data).detach().float()
                         if output.get('overlap_count') is not None else None)
        mask = data['response_mask']
        mask = mask.to_padded_tensor(False) if mask.is_nested else mask
        mask = mask.bool()
        tags = data['opd_source_tag'].reshape(-1)
        require(values.shape == teacher_mass.shape == mask.shape, 'Domain loss/mask shape mismatch')
        require(len(tags) == len(mask), 'Domain tags lost during microbatch splitting')
        require(bool(torch.isfinite(values[mask]).all()) and bool(torch.isfinite(teacher_mass[mask]).all()),
                'Nonfinite valid-token distillation values')
        if student_mass is not None:
            require(student_mass.shape == mask.shape and bool(torch.isfinite(student_mass[mask]).all()),
                    'Student mass lost valid assistant token alignment')
        overlap_fraction = None
        if overlap_count is not None:
            topk = kwargs['distillation_config'].distillation_loss.topk
            require(type(topk) is int and topk > 0, 'Overlap monitoring requires positive Top-K')
            require(overlap_count.shape == mask.shape and bool(torch.isfinite(overlap_count[mask]).all())
                    and bool(((overlap_count[mask] >= 0) & (overlap_count[mask] <= topk)).all()),
                    'Overlap counts lost valid assistant token alignment or exceed Top-K')
            # Match native distillation/overlap_ratio exactly: intersection
            # cardinality divided by K, rather than probability-weighted mass.
            overlap_fraction = overlap_count / topk
        # Native Metric.aggregate_dp averages over attention-DP ranks. Multiply
        # local SUM metrics by DP size so the final values are global sums.
        dp_size = int(data['dp_size'])
        groups = [('opd_domain', SOURCES, tags)]
        if 'opd_teacher_tag' in data:
            names = tuple(kwargs['distillation_config'].teacher_models)
            teacher_tags = data['opd_teacher_tag'].reshape(-1)
            require(len(teacher_tags) == len(mask), 'Teacher tags lost during microbatch splitting')
            groups.append(('opd_teacher', names, teacher_tags))
        for prefix, names, group_tags in groups:
            for code, source in enumerate(names):
                selected = mask & (group_tags == code).to(mask.device).unsqueeze(1)
                lengths = selected.sum(-1)
                per_row = torch.where(selected, values, 0.0).sum(-1)
                stats = dict(loss_sum=per_row.sum(), tokens=lengths.sum(),
                             sequence_mean_sum=(per_row / lengths.clamp_min(1)).sum(),
                             sequences=(lengths > 0).sum(),
                             teacher_mass_sum=torch.where(selected, teacher_mass, 0.0).sum())
                if prefix == 'opd_teacher':
                    if overlap_fraction is not None:
                        stats['overlap_sum'] = torch.where(selected, overlap_fraction, 0.0).sum()
                    if student_mass is not None:
                        stats['student_mass_sum'] = torch.where(selected, student_mass, 0.0).sum()
                for name, value in stats.items():
                    metrics[prefix + '/' + source + '/' + name] = Metric(AggregationType.SUM, value * dp_size)
    return loss, metrics


def cuda_build_env(environ):
    """CUDA build paths required by the supplied Qwen image's Ray recipe."""
    cuda_home = environ.get('CUDA_HOME') or '/usr/local/cuda'
    include_dir = str(Path(cuda_home) / 'include')
    cpath = environ.get('CPATH', '')
    if cpath != include_dir and not cpath.startswith(include_dir + os.pathsep):
        cpath = include_dir + (os.pathsep + cpath if cpath else '')
    updates = {
        'CUDA_HOME': cuda_home,
        'CPATH': cpath,
        'TRITON_PTXAS_PATH': environ.get('TRITON_PTXAS_PATH') or str(Path(cuda_home) / 'bin/ptxas'),
    }
    for key in ('PATH', 'CC', 'C_INCLUDE_PATH', 'LIBRARY_PATH'):
        if environ.get(key):
            updates[key] = environ[key]
    return updates


def check_cuda_build():
    """Check the worker's actual C include search path without using a GPU."""
    import shlex
    import shutil
    import subprocess
    header = Path(os.environ['CUDA_HOME']) / 'include/cuda.h'
    require(header.is_file(), 'CUDA header missing on ' + socket.gethostname() + ': ' + str(header)
            + '; verify CUDA_HOME and the CUDA toolkit mount')
    ptxas = Path(os.environ['TRITON_PTXAS_PATH'])
    require(ptxas.is_file() and os.access(ptxas, os.X_OK), 'ptxas missing or not executable: ' + str(ptxas))
    compiler = shlex.split(os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang') or '')
    require(bool(compiler), 'No C compiler found on ' + socket.gethostname())
    # Deliberately omit -I: compilation must work via the environment inherited
    # by Triton/FLA, not only through a probe-specific include override.
    compiled = subprocess.run(compiler + ['-x', 'c', '-fsyntax-only', '-'],
                              input='#include <cuda.h>\nCUdeviceptr opd_probe_pointer;\n',
                              text=True, capture_output=True, timeout=30)
    require(compiled.returncode == 0, 'CUDA header compile failed on ' + socket.gethostname()
            + ': ' + compiled.stderr[-3000:])
    return dict(cuda_home=os.environ['CUDA_HOME'], cuda_header=str(header),
                cpath=os.environ.get('CPATH', ''), compiler=compiler,
                ptxas=str(ptxas), header_compile_ok=True)
