import copy
from collections import Counter
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opd.online_sampling import OnlineScheduler, EXTERNAL, ROUTES, d3_mixture
from opd.checkpoints import CoverageCheckpointSchedule
from opd.config import dynamic_sampling_budget, normal_sampling_budget, replay_sampling_budget, resource_layout
from opd.common import write_json


class Exp2RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = []
        for route in EXTERNAL:
            for number in range(1180):
                cls.entries.append({'sample_id': '%s-%d' % (route, number),
                                    'teacher_key': route, 'query_fingerprint': '%s-q%d' % (route, number)})
        cls.baseline = list(range(len(cls.entries)))
        for stratum in EXTERNAL:
            for number in range(700):
                cls.entries.append({'sample_id': 'self-%s-%d' % (stratum, number),
                                    'teacher_key': 'self', 'query_fingerprint': 'self-%s-q%d' % (stratum, number),
                                    'sampling_stratum': stratum})

    def settings(self, root, policy='normal', ratio=1):
        options = {'enabled': True, 'epochs': 3, 'external_self_ratio': ratio,
                   'route_sizes': dict.fromkeys(EXTERNAL, 1180), 'fresh_external_queries': 10620,
                   'self_strata_mix': dict(zip(EXTERNAL, (.4, .2, .4)))}
        settings = {'run_dir': str(root), 'seed': 42, 'global_batch_size': 128, 'epochs': 3,
                    'route_loss_coefficients': dict.fromkeys(ROUTES, 1 / 3)}
        if policy == 'normal':
            settings['normal_sampling'] = options
        elif policy == 'replay':
            settings['online_replay'] = dict(options, replay_fraction=.5, warmup_updates=0)
        else:
            settings['dynamic_sampling'] = dict(options, mode='d3_composite', warmup_updates=17,
                update_every=5, window_updates=5, velocity_windows=3, seed_observations=5,
                temperature=.5, ratio_floor=1 / 6, ratio_cap=.5, kl_floor=1e-8)
        return settings

    @staticmethod
    def metrics(step):
        return {f'opd/teacher/{route}/loss': value for route, value in zip(EXTERNAL,
                (max(1., 100. - step * .4), max(.1, 5. - step * .005), .1))}

    def exhaust(self, scheduler, callback=None):
        sizes = []
        while (plan := scheduler.next_plan()) is not None:
            self.assertEqual(len(plan['slots']) % 4, 0)
            self.assertEqual(set(slot['route'] for slot in plan['slots']), set(ROUTES))
            sizes.append(len(plan['slots']))
            scheduler.commit(plan['step'], self.metrics(plan['step']))
            if callback:
                callback(plan)
        self.assertEqual(scheduler.progress_fraction(), 1.)
        return scheduler.outcome(), sizes

    def test_normal_three_complete_epochs_keep_explicit_loss_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = self.settings(temporary)
            settings['route_loss_coefficients'].update(business_reasoning=.5, general=.25, business_agent=.25)
            scheduler = OnlineScheduler(settings, self.entries, self.baseline)
            def coefficients(plan):
                self.assertEqual(plan['coefficients'], settings['route_loss_coefficients'])
            outcome, sizes = self.exhaust(scheduler, coefficients)
        self.assertEqual((outcome['epochs'], outcome['steps'], outcome['exposures']), (3, 166, 21240))
        self.assertEqual(outcome['normal_external_queries'], 10620)
        self.assertEqual(outcome['self_exposures'], 10620)
        self.assertEqual({r: outcome['fresh_by_route'][r] for r in EXTERNAL}, dict.fromkeys(EXTERNAL, 3540))
        self.assertEqual(sizes[-1], 120)
        self.assertEqual(outcome['self_by_stratum'], dict(zip(EXTERNAL, (4248, 2124, 4248))))

    def test_three_to_one_ratio_has_96_external_32_self_without_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = self.settings(temporary, ratio=3)
            scheduler = OnlineScheduler(settings, self.entries, self.baseline)
            counts = []
            def check(plan):
                by_route = Counter(slot['route'] for slot in plan['slots'])
                external = sum(by_route[r] for r in EXTERNAL)
                self.assertEqual(external, 3 * by_route['self'])
                counts.append((external, by_route['self']))
            outcome, sizes = self.exhaust(scheduler, check)
            restored = OnlineScheduler(settings, self.entries, self.baseline)
            restored.load_state_dict(scheduler.state_dict())
        self.assertEqual(counts[0], (96, 32))
        self.assertEqual(counts[-1], (60, 20))
        self.assertEqual((outcome['steps'], outcome['self_exposures'], outcome['exposures']), (111, 3540, 14160))
        self.assertEqual(sizes[-1], 80)
        self.assertIsNone(restored.next_plan())

    def test_replay_no_valid_errors_falls_back_to_normal_and_completes(self):
        with tempfile.TemporaryDirectory() as temporary:
            scheduler = OnlineScheduler(self.settings(temporary, 'replay'), self.entries, self.baseline)
            with mock.patch.object(scheduler, '_error_candidates', return_value=dict.fromkeys(EXTERNAL, [])):
                outcome, _ = self.exhaust(scheduler)
        self.assertEqual((outcome['steps'], outcome['normal_external_queries'], outcome['self_exposures']), (166, 10620, 10620))
        self.assertEqual(outcome['replay_exposures'], 0)

    def test_replay_adds_budget_and_matching_self_but_never_exceeds_half(self):
        with tempfile.TemporaryDirectory() as temporary:
            scheduler = OnlineScheduler(self.settings(temporary, 'replay'), self.entries, self.baseline)
            errors = {route: list(scheduler.original[route]) for route in EXTERNAL}
            def check(plan):
                external = [slot for slot in plan['slots'] if slot['route'] != 'self']
                replay = sum(slot['source'] == 'replay' for slot in external)
                self.assertLessEqual(replay, len(external) / 2)
                self.assertEqual(len(external), sum(slot['route'] == 'self' for slot in plan['slots']))
            with mock.patch.object(scheduler, '_error_candidates', return_value=errors):
                outcome, _ = self.exhaust(scheduler, check)
        self.assertEqual(outcome['normal_external_queries'], 10620)
        self.assertGreater(outcome['replay_exposures'], 0)
        self.assertLessEqual(outcome['replay_exposures'], 10620)
        self.assertEqual(outcome['self_exposures'], 10620 + outcome['replay_exposures'])
        self.assertEqual(outcome['exposures'], 2 * outcome['self_exposures'])

    def test_replay_issued_plan_does_not_change_after_new_feedback_or_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = self.settings(temporary, 'replay')
            scheduler = OnlineScheduler(settings, self.entries, self.baseline)
            with mock.patch.object(scheduler, '_error_candidates', return_value=dict.fromkeys(EXTERNAL, [])):
                first = scheduler.next_plan()
            scheduler.commit(first['step'], {})
            checkpoint = scheduler.state_dict()
            errors = {route: scheduler.original[route][:scheduler.state['cursors'][route]] for route in EXTERNAL}
            with mock.patch.object(scheduler, '_error_candidates', return_value=errors):
                expected = scheduler.next_plan()
            self.assertGreater(expected['diagnostics']['replay_queries'], 0)
            restored = OnlineScheduler(settings, self.entries, self.baseline)
            restored.load_state_dict(checkpoint)
            with mock.patch.object(restored, '_error_candidates', return_value=dict.fromkeys(EXTERNAL, [])) as query:
                self.assertEqual(restored.next_plan(), expected)
            query.assert_not_called()

    def test_dynamic_uses_full_self_and_fixed_budget_then_resumes_issued_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = self.settings(temporary, 'dynamic')
            scheduler = OnlineScheduler(settings, self.entries, self.baseline)
            for _ in range(25):
                plan = scheduler.next_plan()
                if plan['step'] <= 17:
                    self.assertEqual(plan['coefficients'], dict.fromkeys(ROUTES, 1 / 3))
                scheduler.commit(plan['step'], self.metrics(plan['step']))
            checkpoint = scheduler.state_dict()
            expected = scheduler.next_plan()
            restored = OnlineScheduler(settings, self.entries, self.baseline)
            restored.load_state_dict(checkpoint)
            self.assertEqual(restored.next_plan(), expected)
            self.assertEqual(restored.state['history'], checkpoint['history'])
            restored.commit(expected['step'], self.metrics(expected['step']))
            def check(plan):
                coefficients = plan['coefficients']
                self.assertAlmostEqual(sum(coefficients[r] for r in EXTERNAL), 1.)
                self.assertEqual(coefficients['self'], 1 / 3)
                self.assertTrue(all(1 / 6 <= coefficients[r] <= .5 for r in EXTERNAL))
            outcome, _ = self.exhaust(restored, check)
        self.assertEqual((outcome['epochs'], outcome['steps'], outcome['normal_external_queries']), (3, 166, 10620))
        self.assertEqual(outcome['self_exposures'], 10620)
        self.assertEqual(outcome['coverage_mode'], 'fixed_external_budget')
        self.assertNotEqual(len(self.entries), 4720)
        self.assertTrue(any(value != 3540 for r, value in outcome['fresh_by_route'].items() if r in EXTERNAL))

    def test_dynamic_normalizes_each_routes_initial_kl_before_ema(self):
        history = {r: [value[f'opd/teacher/{r}/loss'] for value in (self.metrics(step) for step in range(1, 31))]
                   for r in EXTERNAL}
        scaled = {r: [value * factor for value in history[r]] for r, factor in zip(EXTERNAL, (1000., .01, 50.))}
        first, diagnostics = d3_mixture(history, {'ratio_floor': 1 / 6, 'ratio_cap': .5})
        second, _ = d3_mixture(scaled, {'ratio_floor': 1 / 6, 'ratio_cap': .5})
        for route in EXTERNAL:
            self.assertAlmostEqual(first[route], second[route], places=12)
            self.assertIn('relative_ema', diagnostics['by_route'][route])

    def test_each_policy_saves_exactly_20_coverage_points_including_final(self):
        for policy, ratio in (('normal', 1), ('normal', 3), ('dynamic', 1), ('replay', 1)):
            with self.subTest(policy=policy, ratio=ratio), tempfile.TemporaryDirectory() as temporary:
                scheduler = OnlineScheduler(self.settings(temporary, policy, ratio), self.entries, self.baseline)
                schedule = CoverageCheckpointSchedule({'count': 20, 'start_fraction': .05})
                def save(plan):
                    decision = schedule.decide(plan['step'], scheduler.progress_fraction(), final=plan['last'])
                    if decision:
                        schedule.record_saved(decision)
                errors = {route: list(scheduler.original[route]) for route in EXTERNAL}
                with mock.patch.object(scheduler, '_error_candidates', return_value=errors):
                    outcome, _ = self.exhaust(scheduler, save)
                self.assertEqual(len(schedule.events), 20)
                self.assertEqual(schedule.events[-1]['step'], outcome['steps'])
                self.assertEqual(schedule.events[-1]['reason'], 'final')
                self.assertEqual(set.union(*(set(event['covered_thresholds']) for event in schedule.events)), set(range(20)))

    def test_latest_successful_judge_removes_old_error_and_unjudged_is_not_wrong(self):
        from opd.feedback import result_path
        with tempfile.TemporaryDirectory() as temporary:
            scheduler = OnlineScheduler(self.settings(temporary, 'replay'), self.entries, self.baseline)
            root = Path(temporary) / 'feedback'
            def feedback(query, step, correct, job, status='judged'):
                header = {'job_id': job, 'job_sha256': 'digest', 'step': step,
                          'query_fingerprint': query, 'teacher_key': EXTERNAL[0], 'effective_id': job}
                write_json(root / 'outbox' / (job + '.json'), header)
                write_json(result_path(root, header), {**header, 'correct': correct, 'status': status})
            corrected = self.entries[0]['query_fingerprint']
            feedback(corrected, 2, True, 'new-correct')
            feedback(corrected, 1, False, 'zz-late-old-wrong')
            feedback(self.entries[1]['query_fingerprint'], 2, None, 'unknown')
            feedback(self.entries[2]['query_fingerprint'], 2, False, 'api-error', 'judge_error')
            feedback(self.entries[1100]['query_fingerprint'], 3, False, 'seen-late-in-first-epoch')
            scheduler.state['last_committed_step'] = 3
            scheduler.state['cursors'][EXTERNAL[0]] = 1500
            errors = scheduler._error_candidates()
            self.assertEqual(errors[EXTERNAL[0]], [1100])
            scheduler.state['cursors'][EXTERNAL[0]] = 5
            self.assertEqual(scheduler._error_candidates()[EXTERNAL[0]], [])

    def test_explicit_weights_have_correct_dp4_gradient(self):
        import torch
        from opd.balanced import route_token_weights, weighted_route_loss
        routes = [route for route, count in zip(ROUTES, (21, 21, 22, 64)) for _ in range(count)]
        mask = torch.arange(8).reshape(1, -1) < torch.tensor([1 + i % 8 for i in range(128)]).reshape(-1, 1)
        values = torch.arange(1024, dtype=torch.float64).reshape(128, 8).div(100).requires_grad_()
        coefficients = dict(zip(ROUTES, (.5, .25, .25, 1 / 3)))
        weights = route_token_weights(mask, routes, coefficients, require_equal_queries=False)
        expected = sum(coefficient * values[[r == route for r in routes]][mask[[r == route for r in routes]]].mean()
                       for route, coefficient in coefficients.items())
        actual = sum(weighted_route_loss(value, selected, weight, 4, None)
                     for value, selected, weight in zip(values.chunk(4), mask.chunk(4), weights.chunk(4))) / 4
        self.assertAlmostEqual(actual.detach().item(), expected.detach().item(), places=5)
        self.assertTrue(torch.allclose(torch.autograd.grad(actual, values, retain_graph=True)[0],
                                       torch.autograd.grad(expected, values)[0], atol=1e-8, rtol=1e-5))

    def test_dynamic_config_budget_and_40_48_gpu_resource_layout(self):
        budget = dynamic_sampling_budget(self.settings('/unused', 'dynamic'))
        self.assertEqual(budget['total_exposures'], 21240)
        for nodes, teachers in ((5, 1), (6, 2)):
            self.assertEqual(resource_layout({'nodes': nodes, 'teacher_nodes': teachers,
                              'gpus_per_node': 8, 'train_tp': 8, 'rollout_tp': 2})['actor_gpus'], 32)
        self.assertEqual(normal_sampling_budget(self.settings('/unused', ratio=3))['self_exposures'], 3540)
        replay = replay_sampling_budget(self.settings('/unused', 'replay'))
        self.assertEqual((replay['normal_external_exposures'], replay['maximum_replay_exposures']), (10620, 10620))
        self.assertEqual(math.ceil(replay['total_exposures'] / 128), 332)


if __name__ == '__main__':
    unittest.main()
