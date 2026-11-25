"""CPU value/gradient oracles for query means and asymmetric reverse feedback."""
import ast
from contextlib import contextmanager
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from opd.balanced import route_token_weights, weighted_route_loss
from opd.losses import VocabParallelReverseKL, teacher_partition, token_negative_feedback_alpha
from opd.native import chunked_kl_apply, negative_feedback_topk_loss


@contextmanager
def parallel_state(rank=0, world=1):
    names = ('megatron', 'megatron.core', 'megatron.core.parallel_state')
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    state = ModuleType(names[-1])
    state.get_tensor_model_parallel_group = lambda: None
    state.get_tensor_model_parallel_rank = lambda: rank
    state.get_tensor_model_parallel_world_size = lambda: world
    sys.modules.update({names[0]: ModuleType(names[0]), names[1]: ModuleType(names[1]), names[2]: state})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def fixture():
    generator = torch.Generator().manual_seed(293)
    logits = torch.randn(2, 3, 64, generator=generator, dtype=torch.float64)
    teacher = torch.randn(logits.shape, generator=generator, dtype=torch.float64).log_softmax(-1)
    values, ids = teacher.topk(32, -1)
    weights = torch.randn(2, 3, generator=generator, dtype=torch.float64)
    return logits, values, ids.to(torch.int32), weights


def oracle(logits, teacher_logps, ids, alpha=None, floor=1e-8):
    """Autograd through dense full-vocabulary softmax and one explicit tail."""
    logs = logits.log_softmax(-1)
    probabilities = logs.exp()
    indices = ids.long()
    head = probabilities.gather(-1, indices)
    selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(-1, indices, True)
    tail = probabilities.masked_fill(selected, 0.).sum(-1, keepdim=True)
    student = torch.cat((head, tail), -1)
    target, _ = teacher_partition(teacher_logps, floor)
    ratios = torch.cat((logs.gather(-1, indices), tail.clamp_min(torch.finfo(logits.dtype).tiny).log()), -1)
    ratios = ratios - target.log()
    original = (student * ratios).sum(-1)
    if alpha is None:
        return original
    alpha = token_negative_feedback_alpha(alpha, logits.shape[:-1], logits.device)
    weights = torch.where(ratios > 0, 1-alpha.unsqueeze(-1), 1.)
    transformed = (weights * (student*ratios-student+target)).sum(-1)
    return torch.where(alpha == 0, original, transformed)


def kernel(logits, teacher, ids, alpha=None, floor=1e-8):
    if alpha is None:
        return VocabParallelReverseKL.apply(logits, teacher, ids, None, floor)
    return VocabParallelReverseKL.apply(logits, teacher, ids, None, floor, alpha)


class NegativeFeedbackTests(unittest.TestCase):
    def test_dense_top32_tail_values_gradients_and_recentering(self):
        with parallel_state():
            for alpha in (torch.tensor([.5, 0.], dtype=torch.float64),
                          torch.tensor([[0., .1, .5], [.25, 0., .5]], dtype=torch.float64)):
                logits, teacher, ids, weights = fixture()
                x, y = logits.clone().requires_grad_(), logits.clone().requires_grad_()
                got, expected = kernel(x, teacher, ids, alpha)[0], oracle(y, teacher, ids, alpha)
                (got*weights).sum().backward()
                (expected*weights).sum().backward()
                torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-10)
                torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-10)
                # An unrecentered transformed advantage would fail this test.
                torch.testing.assert_close(x.grad.sum(-1), torch.zeros_like(weights), atol=1e-12, rtol=0)

    def test_zero_alpha_exactly_recovers_original_values_gradients_and_diagnostics(self):
        with parallel_state():
            logits, teacher, ids, weights = fixture()
            x, y = logits.clone().requires_grad_(), logits.clone().requires_grad_()
            original, zero = kernel(x, teacher, ids), kernel(y, teacher, ids, torch.zeros(2, 3))
            for left, right in zip(original, zero):
                self.assertTrue(torch.equal(left, right))
            (original[0]*weights).sum().backward()
            (zero[0]*weights).sum().backward()
            self.assertTrue(torch.equal(x.grad, y.grad))
            mixed = kernel(logits, teacher, ids, torch.tensor([.5, 0.]))
            for left, right in zip(original, mixed):
                self.assertTrue(torch.equal(left[1], right[1]))

    def test_finite_difference_and_checkpointed_alpha_chunks(self):
        with parallel_state():
            logits, teacher, ids, weights = fixture()
            alpha = torch.tensor([[0., .2, .5], [.5, 0., .3]], dtype=torch.float64)
            self.assertTrue(torch.autograd.gradcheck(lambda x: kernel(x, teacher, ids, alpha)[0],
                                                    (logits.clone().requires_grad_(),)))
            for chunk_size in (1, 2, 5):
                x, y = logits.clone().requires_grad_(), logits.clone().requires_grad_()
                apply = lambda z, q, i, clamp, a: VocabParallelReverseKL.apply(z, q, i, clamp, 1e-8, a)
                got = chunked_kl_apply(apply, x, teacher, ids, None, chunk_tokens=chunk_size,
                                       negative_feedback_alpha=alpha)[0]
                expected = oracle(y, teacher, ids, alpha)
                (got*weights).sum().backward()
                (expected*weights).sum().backward()
                torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-10)
                torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-10)

    def test_checkpoint_logits_snapshot_survives_native_in_place_consumption(self):
        with parallel_state():
            logits, teacher, ids, weights = fixture()
            for alpha in (None, torch.tensor([.5, 0.])):
                x = logits.float().requires_grad_()
                y = logits.float().requires_grad_()
                apply = lambda z, q, i, clamp, a=None: VocabParallelReverseKL.apply(z, q, i, clamp, 1e-8, a)
                got = chunked_kl_apply(apply, x, teacher.float(), ids, None, chunk_tokens=2,
                                       negative_feedback_alpha=alpha)[0]
                expected = oracle(y, teacher.float(), ids, alpha)
                with torch.no_grad():
                    x.copy_(torch.full_like(x, -123.))
                (got*weights.float()).sum().backward()
                (expected*weights.float()).sum().backward()
                self.assertIsNotNone(x.grad)
                torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)

    def test_tail_attenuation_and_probability_floor_are_included(self):
        with parallel_state():
            for logit_values, teacher_values in (([0., 0., 4., 3.], [-2., -3.]),
                                                 ([3., 4., -4., -5.], [-.69, -.69]),
                                                 ([0., 0., -1000., -1000.], [-1000., -1000.])):
                x = torch.tensor([[logit_values]], dtype=torch.float64, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                teacher = torch.tensor([[teacher_values]], dtype=torch.float64)
                ids = torch.tensor([[[0, 1]]], dtype=torch.int32)
                got, expected = kernel(x, teacher, ids, torch.tensor(.5), floor=1e-5)[0], \
                    oracle(y, teacher, ids, torch.tensor(.5), floor=1e-5)
                got.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-10)
                torch.testing.assert_close(x.grad, y.grad, atol=1e-12, rtol=1e-10)
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(bool((got >= -1e-12).all()))

    def test_bf16_teacher_overshoot_remains_finite(self):
        with parallel_state():
            logits, _, ids, _ = fixture()
            x = logits.to(torch.bfloat16).requires_grad_()
            teacher = torch.full(ids.shape, math.log(1.0014/32), dtype=torch.bfloat16)
            outputs = kernel(x, teacher, ids, torch.tensor([.5, 0.]))
            outputs[0].sum().backward()
            self.assertTrue(torch.isfinite(outputs[0]).all())
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_two_shard_collectives_match_dense_value_and_local_gradient(self):
        for rank in (0, 1):
            logits, teacher, ids, weights = fixture()
            alpha = torch.tensor([.5, 0.], dtype=torch.float64)
            local = logits[..., rank*32:(rank+1)*32].clone().requires_grad_()
            full = logits.clone().requires_grad_()
            flat = logits.reshape(-1, 64)
            logs = flat.log_softmax(-1)
            probabilities = logs.exp()
            selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter_(-1, ids.reshape(-1, 32).long(), True)
            calls = []
            def all_reduce(value, op=dist.ReduceOp.SUM, group=None):
                calls.append(op)
                expected = (flat.amax(-1),
                            (flat-flat.amax(-1, keepdim=True)).exp().sum(-1),
                            logs.gather(-1, ids.reshape(-1, 32).long()),
                            probabilities.masked_fill(selected, 0.).sum(-1))[len(calls)-1]
                value.copy_(expected)
            def all_gather(outputs, value, group=None):
                for shard in range(2):
                    scores, tokens = logs[:, shard*32:(shard+1)*32].topk(32, -1)
                    outputs[shard].copy_(tokens+shard*32 if value.dtype == torch.int64 else scores)
            with parallel_state(rank, 2), patch.object(dist, 'all_reduce', all_reduce), \
                    patch.object(dist, 'all_gather', all_gather):
                got = kernel(local, teacher, ids, alpha)[0]
            expected = oracle(full, teacher, ids, alpha)
            (got*weights).sum().backward()
            (expected*weights).sum().backward()
            self.assertEqual(len(calls), 4)
            torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-10)
            torch.testing.assert_close(local.grad, full.grad[..., rank*32:(rank+1)*32], atol=1e-12, rtol=1e-10)

    def test_invalid_alpha_rejected(self):
        for alpha in (torch.tensor([-.1, 0.]), torch.tensor([1.1, 0.]),
                      torch.tensor([float('nan'), 0.]), torch.tensor([0, 1]),
                      torch.zeros(3), torch.zeros(2, requires_grad=True)):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                token_negative_feedback_alpha(alpha, (2, 3), torch.device('cpu'))


class QueryMeanTests(unittest.TestCase):
    def fixture(self):
        routes = ['business_reasoning', 'general', 'business_agent', 'self'] * 2
        mask = torch.tensor([[1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                             [1, 0, 1, 0, 1, 1], [1, 0, 0, 0, 0, 0],
                             [1, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 1, 1]], dtype=torch.bool)
        values = torch.arange(48).reshape(8, 6).float()/10
        values[~mask] = float('nan')  # Tool returns/padding never enter means.
        return routes, mask, values

    def test_tool_masked_effective_queries_have_equal_weight_within_each_route(self):
        routes, mask, values = self.fixture()
        coefficients = dict.fromkeys(routes, 1/3)
        weights = route_token_weights(mask, routes, coefficients, aggregation='route_query_mean')
        torch.testing.assert_close(weights*mask.sum(-1), torch.full((8,), 1/6))
        got = weighted_route_loss(values, mask, weights, dp_size=1, max_clamp=2.)
        query_means = torch.stack([values[i, mask[i]].clamp(-2., 2.).mean() for i in range(8)])
        expected = sum(query_means[torch.tensor([r == route for r in routes])].mean()/3
                       for route in coefficients)
        torch.testing.assert_close(got, expected)
        old_weights = route_token_weights(mask, routes, coefficients, aggregation='route')
        self.assertFalse(torch.allclose(weights, old_weights))

    def test_dp_and_microbatch_partition_invariance_values_and_gradients(self):
        routes, mask, initial = self.fixture()
        weights = route_token_weights(mask, routes, dict.fromkeys(routes, 1/3), aggregation='route_query_mean')
        x, y = initial.clone().requires_grad_(), initial.clone().requires_grad_()
        full = weighted_route_loss(x, mask, weights, 1, max_clamp=2.)
        # Uneven route/length composition across ranks; DP averages gradients.
        partitions = ((0, 3, 5, 6), (1, 2, 4, 7))
        split = sum(sum(weighted_route_loss(y[i:i+1], mask[i:i+1], weights[i:i+1], 2, max_clamp=2.)
                        for i in rank) for rank in partitions)/2
        full.backward()
        split.backward()
        torch.testing.assert_close(full, split)
        torch.testing.assert_close(x.grad, y.grad)
        self.assertEqual(x.grad[~mask].abs().sum().item(), 0.)
        self.assertEqual(x.grad[mask & (initial > 2.)].abs().sum().item(), 0.)

    def test_empty_query_rejected_even_when_its_route_has_other_tokens(self):
        routes, mask, _ = self.fixture()
        mask[0] = False
        with self.assertRaisesRegex(ValueError, 'Empty assistant mask'):
            route_token_weights(mask, routes, aggregation='route_query_mean')
        # The route-token baseline retains its previous valid behavior.
        route_token_weights(mask, routes, aggregation='route')

    def test_dynamic_external_coefficients_and_unequal_query_counts_preserve_joint_weights(self):
        routes = ['business_reasoning']*3 + ['general']*2 + ['business_agent', 'self']
        mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 0],
                             [1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        route_values = dict(business_reasoning=1., general=2., business_agent=3., self=4.)
        for aggregation in ('route', 'route_query_mean'):
            for external in ((.5, .3, .2), (.2, .5, .3)):
                coefficients = dict(zip(('business_reasoning', 'general', 'business_agent'), external))
                coefficients['self'] = 1/3
                values = torch.tensor([[route_values[route]]*4 for route in routes], requires_grad=True)
                weights = route_token_weights(mask, routes, coefficients, require_equal_queries=False,
                                              aggregation=aggregation)
                loss = weighted_route_loss(values, mask, weights, 1, max_clamp=None)
                expected = sum(coefficients[r]*route_values[r] for r in coefficients)
                torch.testing.assert_close(loss, torch.tensor(expected))
                loss.backward()
                for route, coefficient in coefficients.items():
                    rows = torch.tensor([r == route for r in routes])
                    torch.testing.assert_close(values.grad[rows].sum(), torch.tensor(coefficient))


class NativePackingTests(unittest.TestCase):
    def test_actual_native_thd_bshd_cp_packing_keeps_alpha_with_its_teacher_row(self):
        source = Path(os.environ.get('KAI_VERL_ROOT', str(ROOT.parent / 'kai-verl'))) / 'verl/models/mcore/util.py'
        if not source.is_file():
            self.skipTest('Supplied native source checkout is unavailable')
        tree = ast.parse(source.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('preprocess_thd_engine', 'preprocess_bshd_engine')]
        for cp_rank in (0, 1):
            mpu = SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 2,
                                  get_context_parallel_world_size=lambda: 2,
                                  get_context_parallel_rank=lambda: cp_rank)
            scope = dict(torch=torch, math=math, mpu=mpu, is_npu_available=False,
                         preprocess_for_mindspeed=lambda *args: None, PackedSeqParams=SimpleNamespace)
            module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                                     *functions], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), '<actual native preprocessing>', 'exec'), scope)
            util = ModuleType('verl.models.mcore.util')
            util.preprocess_thd_engine, util.preprocess_bshd_engine = \
                scope['preprocess_thd_engine'], scope['preprocess_bshd_engine']
            native = ModuleType('verl.trainer.distillation.megatron.losses')
            captured = []
            def apply(logits, teacher, ids, clamp, alpha):
                captured.append((ids.clone(), alpha.clone()))
                return tuple(torch.zeros(logits.shape[:2]) for _ in range(5))
            native._VocabParallelKLDivergence = SimpleNamespace(apply=apply)
            names = ('verl', 'verl.models', 'verl.models.mcore', 'verl.trainer', 'verl.trainer.distillation',
                     'verl.trainer.distillation.megatron')
            modules = {name: ModuleType(name) for name in names}
            modules.update({util.__name__: util, native.__name__: native})
            offsets = torch.tensor([0, 8, 20])
            logps = torch.nested.nested_tensor_from_jagged(torch.full((20, 2), -2.), offsets=offsets)
            ids = torch.nested.nested_tensor_from_jagged(
                torch.cat((torch.full((8, 2), 9), torch.full((12, 2), 11))), offsets=offsets)
            data = {'teacher_logprobs': logps, 'teacher_ids': ids, 'opd_negative_alpha': torch.tensor([.5, 0.])}
            config = SimpleNamespace(strategy='megatron')
            distillation = SimpleNamespace(distillation_loss=SimpleNamespace(
                loss_mode='reverse_kl_topk_tail', log_prob_min_clamp=None))
            with patch.dict(sys.modules, modules):
                for format_name, preprocess in (('thd', util.preprocess_thd_engine),
                                                ('bshd', util.preprocess_bshd_engine)):
                    packed_ids = preprocess(ids)[0]
                    logits = torch.zeros((*packed_ids.shape[:2], 32))
                    outputs = negative_feedback_topk_loss(logits, data, distillation, format_name, config)
                    self.assertEqual(tuple(outputs), ('distillation_losses', 'student_mass', 'teacher_mass',
                                                     'overlap_count', 'overlap_token_advantage'))
                    got_ids, alpha = captured[-1]
                    torch.testing.assert_close(alpha[got_ids[..., 0] == 9],
                                               torch.full_like(alpha[got_ids[..., 0] == 9], .5))
                    self.assertEqual(alpha[got_ids[..., 0] != 9].abs().sum().item(), 0.)
                    self.assertEqual(alpha.shape, logits.shape[:2])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
