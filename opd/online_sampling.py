"""Restartable, batch-boundary query selection; no GPU or network dependency.

D3-composite is adapted to our explicit route means. External loss coefficients
follow the same frozen batch mixture used for draws; self remains 1/3.
"""
from collections import Counter
import copy
import math
from pathlib import Path
import random

from opd.common import fingerprint, read_json, require, write_json
from opd.sampling import identity

EXTERNAL = ('business_reasoning', 'general', 'business_agent')
ROUTES = EXTERNAL + ('self',)
CORE_FIELDS = ('next_step', 'cursors', 'self_cursor', 'fresh_groups', 'replay_rotors',
               'fresh_by_route', 'replay_by_route', 'exposures_by_route', 'exposures',
               'complete', 'last_plan_step')
COMBO_FIELDS = CORE_FIELDS + ('self_strata_cursors', 'self_by_stratum')


def is_online(settings):
    return bool(settings.get('normal_sampling', {}).get('enabled')
                or settings.get('online_replay', {}).get('enabled')
                or settings.get('dynamic_sampling', {}).get('mode') == 'd3_composite')


def ema(values):
    require(bool(values), 'Empty KL smoothing window')
    alpha = 2.0 / (len(values) + 1)
    value = float(values[0])
    for observation in values[1:]:
        value = alpha * float(observation) + (1 - alpha) * value
    return value


def bounded_mixture(weights, floor, cap):
    """Redistribute clipped softmax mass inside a feasible simplex box."""
    require(len(weights) * floor <= 1 <= len(weights) * cap, 'Infeasible route bounds')
    lo, hi = 0., 1. / min(weights.values())
    for _ in range(80):
        scale = (lo + hi) / 2
        total = sum(min(cap, max(floor, scale * p)) for p in weights.values())
        if total < 1:
            lo = scale
        else:
            hi = scale
    result = {r: min(cap, max(floor, (lo + hi) / 2 * p)) for r, p in weights.items()}
    require(math.isclose(sum(result.values()), 1., abs_tol=1e-12), 'Bounded route weights lost mass')
    return result


def d3_mixture(history, settings):
    """Official product signal, using adjacent nonoverlapping EMA windows.

    A short run uses W=5/update=5 rather than the paper's W=10/update=10.
    KL floor is a numerical epsilon appropriate to our dense bucket-KL scale;
    it is not the paper's absolute 0.15 floor. Rehearsal/self is excluded.
    """
    window = int(settings.get('window_updates', 5))
    seed_count = int(settings.get('seed_observations', 5))
    windows = int(settings.get('velocity_windows', 3))
    temperature = float(settings.get('temperature', .5))
    floor = float(settings.get('ratio_floor', .2))
    cap = float(settings.get('ratio_cap') or 1.)
    kl_floor = float(settings.get('kl_floor', 1e-8))
    require(window > 0 and seed_count > 0 and windows > 0 and temperature > 0
            and 0 <= floor < 1 / 3 <= cap <= 1 and kl_floor > 0, 'Invalid D3 scheduler hyperparameters')
    uniform = dict.fromkeys(EXTERNAL, 1 / 3)
    if any(len(history.get(r, [])) < max(seed_count, 2 * window) for r in EXTERNAL):
        return uniform, {'state': 'warmup', 'signal': dict.fromkeys(EXTERNAL, 0.)}
    signals, diagnostics = {}, {}
    for route in EXTERNAL:
        values = [max(0., float(v)) for v in history[route]]
        require(all(math.isfinite(v) for v in values), 'Nonfinite D3 observations')
        initial = sum(values[:seed_count]) / seed_count
        # Normalize each teacher against its own initial KL BEFORE smoothing.
        # Absolute KL scales cannot make one domain dominate another.
        relative = [value / max(initial, kl_floor) for value in values]
        number = min(windows, len(values) // window - 1)
        snapshots = [ema(relative[len(relative) - (i + 1) * window:len(relative) - i * window])
                     for i in range(number + 1)]
        deltas = [(snapshots[i] - snapshots[i + 1]) / max(snapshots[i + 1], kl_floor)
                  for i in range(number)]
        velocity = max(0., -sum(deltas) / number)
        gap = snapshots[0]
        # Numerical cancellation around zero must not create artificial urgency.
        signal = gap * velocity if initial > kl_floor else 0.
        signals[route] = signal
        diagnostics[route] = {'initial_kl': initial, 'ema': snapshots[0] * initial,
                              'relative_ema': snapshots[0], 'gap': gap,
                              'velocity': velocity, 'relative_changes': deltas, 'signal': signal}
    maximum = max(signals.values())
    normalized = {r: signals[r] / maximum if maximum > 0 else 0. for r in EXTERNAL}
    peak = max(normalized.values()) / temperature
    weights = {r: math.exp(normalized[r] / temperature - peak) for r in EXTERNAL}
    total = sum(weights.values())
    ratios = {r: floor + (1 - 3 * floor) * weights[r] / total for r in EXTERNAL}
    ratios = bounded_mixture(ratios, floor, cap)
    return ratios, {'state': 'active' if maximum > 0 else 'all_plateau_uniform',
                    'by_route': diagnostics, 'normalized_signal': normalized, 'mixture': ratios}


def largest_remainder(total, ratios, floor=0., cap=1.):
    require(type(total) is int and total >= len(ratios), 'Too few external slots')
    require(all(math.isfinite(v) and v > 0 for v in ratios.values()), 'Invalid sampling mixture')
    weight = sum(ratios.values())
    exact = {r: total * p / weight for r, p in ratios.items()}
    counts = {r: math.floor(v) for r, v in exact.items()}
    for route in sorted(exact, key=lambda r: (-(exact[r] - counts[r]), r))[:total - sum(counts.values())]:
        counts[route] += 1
    lower, upper = math.ceil(total * floor), math.floor(total * cap)
    require(len(ratios) * lower <= total <= len(ratios) * upper, 'Integer route limits infeasible')
    counts = {r: min(upper, max(lower, n)) for r, n in counts.items()}
    while sum(counts.values()) < total:
        route = max((r for r in counts if counts[r] < upper), key=lambda r: (exact[r] - counts[r], r))
        counts[route] += 1
    while sum(counts.values()) > total:
        route = max((r for r in counts if counts[r] > lower), key=lambda r: (counts[r] - exact[r], r))
        counts[route] -= 1
    require(sum(counts.values()) == total and all(counts.values()), 'Quota rounding lost a route')
    return counts


class OnlineScheduler:
    """Persist every issued batch before generation; checkpoint consumed cursors.

    Existing issued plans are reused after an interrupted optimizer/checkpoint,
    so new Judge completions never alter a batch that was already issued.
    """
    def __init__(self, settings, entries, baseline_order):
        self.settings = dict(settings)
        self.entries = entries
        self.run = Path(settings['run_dir'])
        self.plan_root = self.run / 'sampling/plans'
        self.batch_size = int(settings['global_batch_size'])
        require(self.batch_size == 128, 'Stateful combination sampling currently requires BS128')
        self.normal = bool(settings.get('normal_sampling', {}).get('enabled'))
        self.dynamic = settings.get('dynamic_sampling', {}).get('mode') == 'd3_composite'
        self.token_budget = bool(settings.get('token_budget_sampling', {}).get('enabled'))
        require(not self.token_budget or (settings.get('online_replay', {}).get('enabled')
                                         and not self.normal and not self.dynamic),
                'Token-budget sampling requires the combination error-replay path')
        require(not self.normal or not (settings.get('online_replay', {}).get('enabled')
                                       or settings.get('dynamic_sampling', {}).get('mode') == 'd3_composite'),
                'Normal sampling cannot enable error replay or dynamic sampling')
        self.combo = self.normal or 'epochs' in settings.get('online_replay', {}) or (self.dynamic and 'epochs' in settings['dynamic_sampling'])
        require(not self.token_budget or self.combo, 'Token-budget replay needs an explicit multi-epoch normal budget')
        if self.combo:
            self._init_combo(baseline_order)
            return
        require(len(baseline_order) == 4720 and len(set(baseline_order)) == 4720,
                'Online sampling requires original 4720-query baseline order')
        self.baseline = list(baseline_order)
        self.original = {r: [i for i in self.baseline if entries[i]['teacher_key'] == r] for r in ROUTES}
        require(all(len(v) == 1180 for v in self.original.values()), 'Original four-route data changed')
        self.policy = 'error_replay' if settings.get('online_replay', {}).get('enabled') else 'd3_composite'
        require(self.policy == 'error_replay' or settings.get('dynamic_sampling', {}).get('mode') == 'd3_composite',
                'Unknown online scheduler')
        self.pool = copy.deepcopy(self.original)
        if self.policy == 'd3_composite':
            used = set(self.baseline)
            for route in EXTERNAL:
                extras = [i for i, e in enumerate(entries) if i not in used and e['teacher_key'] == route]
                random.Random(int(settings['seed']) + ROUTES.index(route)).shuffle(extras)
                self.pool[route].extend(extras)
        self.query_lookup = {entries[i]['query_fingerprint']: i for i in self.baseline}
        self.contract = fingerprint({'policy': self.policy, 'entries': identity(entries), 'baseline': self.baseline,
                                     'batch_size': self.batch_size, 'seed': settings['seed'],
                                     'online_replay': settings.get('online_replay'),
                                     'dynamic_sampling': settings.get('dynamic_sampling')})
        self.state = {'contract': self.contract, 'next_step': 1, 'cursors': dict.fromkeys(EXTERNAL, 0),
                      'self_cursor': 0, 'fresh_groups': 0, 'replay_rotors': dict.fromkeys(EXTERNAL, 0),
                      'fresh_by_route': dict.fromkeys(ROUTES, 0), 'replay_by_route': dict.fromkeys(ROUTES, 0),
                      'exposures_by_route': dict.fromkeys(ROUTES, 0), 'exposures': 0, 'complete': False,
                      'last_plan_step': 0, 'last_committed_step': 0, 'pending_batch': None,
                      'history': dict.fromkeys(EXTERNAL, None), 'mixture': dict.fromkeys(EXTERNAL, 1 / 3),
                      'mixture_diagnostics': {'state': 'warmup'}}
        self.state['history'] = {r: [] for r in EXTERNAL}
        self.last_plan = None

    def _init_combo(self, baseline_order):
        """A normal epoch visits every external candidate once.

        The seed order contains one copy of the selected external candidates.
        Self draws use their complete independent pool and match *all* external
        draws, including replay. Reusing a query never reuses its trajectory.
        """
        option_key = 'normal_sampling' if self.normal else 'dynamic_sampling' if self.dynamic else 'online_replay'
        options = self.settings[option_key]
        require(options.get('enabled') is True or self.dynamic, 'Combination sampler must be enabled')
        self.epochs = int(options['epochs'])
        self.external_self_ratio = float(options.get('external_self_ratio', 1))
        require(self.epochs > 0 and self.external_self_ratio in ((1., 3.) if self.normal else (1.,)),
                'Combination sampler requires positive epochs and a supported external:self ratio')
        if not self.normal and not self.dynamic:
            require(math.isclose(float(options.get('replay_fraction', .5)), .5),
                    'Combination replay target must be 50% within each external route')
        require(isinstance(baseline_order, (list, tuple)) and baseline_order
                and all(type(i) is int and 0 <= i < len(self.entries) for i in baseline_order)
                and len(set(baseline_order)) == len(baseline_order),
                'Combination seed must contain distinct valid candidate indices')
        self.baseline = list(baseline_order)
        self.original = {r: [i for i in self.baseline if self.entries[i]['teacher_key'] == r]
                         for r in EXTERNAL}
        self.original['self'] = [i for i, e in enumerate(self.entries) if e['teacher_key'] == 'self']
        require(all(self.original.values()), 'Combination data requires all three external routes and self')
        self.route_sizes = {r: len(self.original[r]) for r in EXTERNAL}
        require(options.get('route_sizes', self.route_sizes) == self.route_sizes,
                'Combination frozen route sizes differ from replay settings')
        self.normal_external_budget = sum(self.route_sizes.values()) * self.epochs
        require((self.normal_external_budget / self.external_self_ratio).is_integer(),
                'External:self ratio needs an integer self budget')
        require(self.normal_external_budget % 2 == 0,
                'External normal budget must be even for an exact 1:1 DP4 tail')
        require(options.get('fresh_external_queries', self.normal_external_budget) == self.normal_external_budget,
                'Fresh external query budget must equal all candidates times epochs')
        self.route_budgets = {r: size * self.epochs for r, size in self.route_sizes.items()}
        # Each normal batch consumes 64 external queries; legacy replay batches
        # consume at least 32. Every route must support this update-count bound.
        self.external_batch_capacity = int(self.batch_size * self.external_self_ratio / (1 + self.external_self_ratio))
        minimum_fresh_batch = self.external_batch_capacity if self.normal or self.dynamic else 32
        require(min(self.route_budgets.values()) >= math.ceil(self.normal_external_budget / minimum_fresh_batch),
                'Route pool is too small to appear in every optimizer batch without padding')
        self.policy = 'normal_epochs' if self.normal else 'd3_composite' if self.dynamic else 'error_replay'
        self.pool = copy.deepcopy(self.original)
        random.Random(f"{self.settings['seed']}:combo:self").shuffle(self.pool['self'])
        mix = options.get('self_strata_mix')
        require(isinstance(mix, dict) and set(mix) == set(EXTERNAL)
                and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in mix.values()),
                'Self draws require the original three-stratum exposure proportions')
        self.self_mix = {r: mix[r] / sum(mix.values()) for r in EXTERNAL}
        self.self_pools = {r: [i for i in self.pool['self'] if self.entries[i].get('sampling_stratum') == r]
                           for r in EXTERNAL}
        require(sum(map(len, self.self_pools.values())) == len(self.pool['self'])
                and all(len(v) >= 64 for v in self.self_pools.values()),
                'Full self pool requires three valid strata, each at least one batch wide')
        self.query_lookup = {}
        self.route_query_lookup = {}
        for route in EXTERNAL:
            for index in self.original[route]:
                query = self.entries[index]['query_fingerprint']
                require((route, query) not in self.route_query_lookup,
                        'Duplicate question within one external teacher route')
                self.route_query_lookup[route, query] = index
                self.query_lookup.setdefault(query, index)
        require(len({self.entries[i]['query_fingerprint'] for i in self.pool['self']}) == len(self.pool['self']),
                'Self pool must be deduplicated within its route')
        contract = {'schema': 'mopd_combo_normal_v2' if self.normal else 'mopd_combo_dynamic_v2' if self.dynamic else 'mopd_combo_replay_v2',
                                     'policy': self.policy,
                                     'entries': identity(self.entries), 'baseline': self.baseline,
                                     'batch_size': self.batch_size, 'seed': self.settings['seed'],
                                     option_key: options,
                                     'route_loss_coefficients': self.settings.get('route_loss_coefficients')}
        if self.token_budget:
            contract['token_budget_sampling'] = self.settings['token_budget_sampling']
        self.contract = fingerprint(contract)
        self.state = {'contract': self.contract, 'next_step': 1, 'cursors': dict.fromkeys(EXTERNAL, 0),
                      'self_cursor': 0, 'fresh_groups': 0, 'replay_rotors': dict.fromkeys(EXTERNAL, 0),
                      'fresh_by_route': dict.fromkeys(ROUTES, 0), 'replay_by_route': dict.fromkeys(ROUTES, 0),
                      'exposures_by_route': dict.fromkeys(ROUTES, 0), 'exposures': 0, 'complete': False,
                      'last_plan_step': 0, 'last_committed_step': 0, 'pending_batch': None,
                      'history': {r: [] for r in EXTERNAL}, 'mixture': dict.fromkeys(EXTERNAL, 1 / 3),
                      'mixture_diagnostics': {'state': 'fixed_route_means'}}
        self.state.update(self_strata_cursors=dict.fromkeys(EXTERNAL, 0),
                          self_by_stratum=dict.fromkeys(EXTERNAL, 0))
        if self.token_budget:
            from opd.token_budget import initial_state
            self.state['token_sampling'] = initial_state(self.settings['token_budget_sampling'])
            self.state['coverage_mode'] = 'fixed_external_budget'
        if self.dynamic:
            self.state['mixture_diagnostics'] = {'state': 'warmup'}
        self.last_plan = None

    @property
    def core_fields(self):
        fields = COMBO_FIELDS if self.combo else CORE_FIELDS
        return fields + ('token_sampling', 'coverage_mode') if self.token_budget else fields

    @property
    def maximum_updates(self):
        # A bound for the framework progress display, not a stopping condition.
        # At most one replay exposure per fresh external exposure is scheduled.
        if self.combo:
            external = self.normal_external_budget * (1 if self.normal or self.dynamic else 2)
            return math.ceil(external * (1 + 1 / self.external_self_ratio) / self.batch_size)
        return math.ceil((4720 + 3540) / self.batch_size) if self.policy == 'error_replay' else 37

    def progress_fraction(self):
        """Committed normal coverage, excluding error replay and self draws."""
        counts = self.state['fresh_by_route']
        if self.state['pending_batch'] is not None:
            require(self.last_plan is not None, 'Pending sampling plan is missing')
            counts = self.last_plan['before']['fresh_by_route']
        budget = self.normal_external_budget if self.combo else 3540
        return min(1., sum(counts[r] for r in EXTERNAL) / budget)

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        require(state.get('contract') == self.contract, 'Online sampling identity differs from checkpoint')
        require(state.get('pending_batch') is None, 'Checkpoint was saved before completing its optimizer batch')
        if self.combo:
            require(all(type(state['cursors'][r]) is int and 0 <= state['cursors'][r]
                        and (self.dynamic or self.token_budget or state['cursors'][r] <= self.route_budgets[r])
                        and state['fresh_by_route'][r] == state['cursors'][r] for r in EXTERNAL),
                    'Checkpoint normal coverage counters differ')
            require(all(type(state['replay_by_route'][r]) is int and state['replay_by_route'][r] >= 0
                        and state['exposures_by_route'][r] == state['fresh_by_route'][r] + state['replay_by_route'][r]
                        for r in EXTERNAL) and state['replay_by_route']['self'] == 0,
                    'Checkpoint route exposure counters differ')
            if self.normal or self.dynamic:
                require(all(state['replay_by_route'][r] == 0 for r in ROUTES),
                        'Normal sampling checkpoint contains replay exposures')
            external = sum(state['exposures_by_route'][r] for r in EXTERNAL)
            self_exposures = external / self.external_self_ratio
            require(state['self_cursor'] == state['fresh_by_route']['self'] == self_exposures
                    and state['exposures_by_route']['self'] == self_exposures
                    and state['exposures'] == external + self_exposures,
                    'Checkpoint external:self exposure ratio differs')
            require(state['self_strata_cursors'] == state['self_by_stratum']
                    and sum(state['self_by_stratum'].values()) == self_exposures,
                    'Checkpoint self stratum exposure counters differ')
            require(state['last_committed_step'] == state['last_plan_step']
                    and state['next_step'] == state['last_committed_step'] + 1,
                    'Checkpoint sampling/optimizer step counters differ')
            exhausted = (sum(state['cursors'].values()) == self.normal_external_budget if self.dynamic or self.token_budget
                         else all(state['cursors'][r] == self.route_budgets[r] for r in EXTERNAL))
            require(state['complete'] == exhausted,
                    'Checkpoint completion differs from normal epoch coverage')
            if self.token_budget:
                from opd.token_budget import validate
                validate(self.settings['token_budget_sampling'])
                token = state['token_sampling']
                require(state.get('coverage_mode') == 'fixed_external_budget'
                        and sum(state['cursors'].values()) <= self.normal_external_budget
                        and all(state['replay_by_route'][r] <= state['fresh_by_route'][r] for r in EXTERNAL),
                        'Token checkpoint exceeded its normal/replay exposure budget')
                require(all(token['queries_by_route'][r] == state['exposures_by_route'][r]
                            and token['queries_by_source']['fresh'][r] == state['fresh_by_route'][r]
                            and token['queries_by_source']['replay'][r] == state['replay_by_route'][r]
                            and math.isfinite(token['tokens_by_route'][r]) and token['tokens_by_route'][r] >= 0
                            and (token['length_ema'][r] is None or
                                 math.isfinite(token['length_ema'][r]) and token['length_ema'][r] > 0)
                            for r in EXTERNAL), 'Token checkpoint observations differ from committed exposures')
            if self.dynamic:
                require(sum(state['cursors'].values()) <= self.normal_external_budget,
                        'Dynamic checkpoint exceeded its external exposure budget')
                require(all(len(state['history'][route]) == state['last_committed_step']
                            and all(math.isfinite(value) and value >= 0 for value in state['history'][route])
                            for route in EXTERNAL), 'Dynamic KL history differs from committed updates')
        self.state = copy.deepcopy(state)
        self.last_plan = None

    def _error_candidates(self):
        require(not self.normal, 'Normal sampling does not read error feedback')
        from opd.feedback import successful_feedback
        successful = successful_feedback(self.run, self.state['last_committed_step'])
        if self.combo:
            # A route consumes its original unique order before wrapping. This
            # prefix therefore covers all seen questions, including later epochs.
            # successful_feedback already chooses the newest committed boolean
            # verdict; late older jobs cannot re-introduce a corrected mistake.
            consumed = {(r, self.entries[i]['query_fingerprint']) for r in EXTERNAL
                        for i in self.original[r][:min(self.state['cursors'][r], self.route_sizes[r])]}
            errors = {r: [] for r in EXTERNAL}
            for verdict in successful:
                key = (verdict.get('teacher_key'), verdict.get('query_fingerprint'))
                if verdict.get('correct') is False and key in consumed and key in self.route_query_lookup:
                    errors[key[0]].append(self.route_query_lookup[key])
            return {r: sorted(set(indices)) for r, indices in errors.items()}
        consumed = {self.entries[i]['query_fingerprint'] for r in EXTERNAL
                    for i in self.original[r][:self.state['cursors'][r]]}
        errors = {r: [] for r in EXTERNAL}
        for verdict in successful:
            query = verdict.get('query_fingerprint')
            if verdict.get('correct') is not False or query not in consumed or query not in self.query_lookup:
                continue
            index = self.query_lookup[query]
            route = self.entries[index]['teacher_key']
            if route in errors:
                errors[route].append(index)
        for route in errors:
            errors[route] = sorted(set(errors[route]))
        return errors

    def _replay_plan(self, step):
        if self.combo:
            return self._combo_token_replay_plan(step) if self.token_budget else self._combo_replay_plan(step)
        options = self.settings['online_replay']
        require(math.isclose(float(options.get('replay_fraction', .5)), .5), 'Replay target must be 50% of external exposures')
        remaining_groups = 1180 - self.state['fresh_groups']
        require(remaining_groups > 0, 'Replay scheduler has no fresh queries')
        errors = dict.fromkeys(EXTERNAL, [])
        error_problem = None
        if step > int(options.get('warmup_updates', 10)):
            try:
                errors = self._error_candidates()
            except Exception as exc:
                # Scheduler failure must degrade to fresh sampling, not halt training.
                error_problem = f'{type(exc).__name__}: {exc}'
                print('Online Judge feedback unavailable; use fresh queries: ' + error_problem, flush=True)
        # Fresh groups always contain one query from each external route + one
        # self. This guarantees 3540 fresh external and 1180 self, each once.
        # All normal batches are BS128; replay deficits fill with fresh groups.
        max_fresh = min(self.batch_size // 4, remaining_groups)
        chosen = None
        for groups in range(math.ceil(self.batch_size / 7), max_fresh + 1):
            replay_needed = self.batch_size - 4 * groups
            # Keep replay <= fresh external in this batch. Do not compensate
            # warmup by abruptly issuing a replay-dominated batch.
            replay_limit = 3 * groups
            capacity = sum(min(groups, len(v)) for v in errors.values())
            if replay_needed <= replay_limit and replay_needed <= capacity:
                chosen = (groups, replay_needed)
                break
        if chosen is None:
            groups = max_fresh
            # A smaller final batch is rounded down only by reducing optional
            # replay, never by dropping/padding any fresh or self query.
            replay = min(self.batch_size - 4 * groups,
                         3 * groups,
                         sum(min(groups, len(v)) for v in errors.values()))
            replay -= replay % 4
        else:
            groups, replay = chosen
        slots = []
        for _ in range(groups):
            for route in EXTERNAL:
                index = self.original[route][self.state['cursors'][route]]
                self.state['cursors'][route] += 1
                slots.append({'index': index, 'source': 'fresh', 'route': route})
            index = self.original['self'][self.state['self_cursor']]
            self.state['self_cursor'] += 1
            slots.append({'index': index, 'source': 'fresh', 'route': 'self'})
        self.state['fresh_groups'] += groups
        # Fair round-robin allocation of optional replay among available routes.
        selected, used = dict.fromkeys(EXTERNAL, 0), set()
        for count in range(replay):
            available = [r for r in EXTERNAL if selected[r] < min(groups, len(errors[r]))]
            route = min(available, key=lambda r: (selected[r], EXTERNAL.index(r)))
            pool = errors[route]
            cursor = self.state['replay_rotors'][route]
            index = pool[cursor % len(pool)]
            while index in used:
                cursor += 1
                index = pool[cursor % len(pool)]
            self.state['replay_rotors'][route] = cursor + 1
            selected[route] += 1
            used.add(index)
            slots.append({'index': index, 'source': 'replay', 'route': route})
        random.Random(int(self.settings['seed']) + step).shuffle(slots)
        return slots, {'error_pool_sizes': {r: len(errors[r]) for r in EXTERNAL},
                       'feedback_problem': error_problem, 'external_replay_target': .5,
                       'fresh_groups': groups, 'replay_queries': replay}, self._fixed_coefficients()

    def _combo_fresh_counts(self, total, remaining):
        """Natural remaining mix, reserving enough of every route for later batches."""
        left = sum(remaining.values()) - total
        minimum_fresh_batch = self.external_batch_capacity if self.normal or self.dynamic else 32
        reserve = math.ceil(left / minimum_fresh_batch) if left else 0
        maximum = {r: remaining[r] - reserve for r in EXTERNAL}
        if any(capacity < 1 for capacity in maximum.values()) or sum(maximum.values()) < total:
            return None
        counts = dict.fromkeys(EXTERNAL, 1)
        denominator = sum(remaining.values())
        for _ in range(total - len(EXTERNAL)):
            route = max((r for r in EXTERNAL if counts[r] < maximum[r]),
                        key=lambda r: (total * remaining[r] / denominator - counts[r], -EXTERNAL.index(r)))
            counts[route] += 1
        return counts

    def _combo_fresh_indices(self, counts):
        return {r: [self.original[r][(self.state['cursors'][r] + offset) % self.route_sizes[r]]
                    for offset in range(counts[r])] for r in EXTERNAL}

    def _combo_self_draws(self, count):
        slots = []
        for _ in range(count):
            number = self.state['self_cursor'] + 1
            route = max(EXTERNAL, key=lambda r: (number * self.self_mix[r] - self.state['self_by_stratum'][r],
                                                -EXTERNAL.index(r)))
            cursor = self.state['self_strata_cursors'][route]
            index = self.self_pools[route][cursor % len(self.self_pools[route])]
            self.state['self_strata_cursors'][route] += 1
            self.state['self_by_stratum'][route] += 1
            self.state['self_cursor'] += 1
            slots.append({'index': index, 'source': 'fresh', 'route': 'self', 'stratum': route})
        return slots

    def _combo_replay_plan(self, step):
        require(not self.normal, 'Normal sampling cannot schedule error replay')
        options = self.settings['online_replay']
        remaining = {r: self.route_budgets[r] - self.state['cursors'][r] for r in EXTERNAL}
        normal_left = sum(remaining.values())
        require(normal_left > 0 and all(remaining.values()), 'Normal epoch route budget exhausted prematurely')
        errors = {r: [] for r in EXTERNAL}
        error_problem = None
        if step > int(options.get('warmup_updates', 10)):
            try:
                errors = self._error_candidates()
            except Exception as exc:
                error_problem = f'{type(exc).__name__}: {exc}'
                print('Online Judge feedback unavailable; use fresh queries: ' + error_problem, flush=True)
        # Both normal and replay totals are even. Thus external:self=1:1
        # always makes a DP4-divisible batch without discarding or padding data.
        # A reserve prevents a low-count route disappearing near the tail.
        candidates = []
        lower = 32 if normal_left > 32 else normal_left
        fresh_totals = range(lower, min(64, normal_left) + 1, 2)
        no_errors = not any(errors.values())
        if no_errors:
            fresh_totals = reversed(fresh_totals)
        for fresh_total in fresh_totals:
            counts = self._combo_fresh_counts(fresh_total, remaining)
            if counts is None:
                continue
            indices = self._combo_fresh_indices(counts)
            fresh_sets = {r: set(indices[r]) for r in EXTERNAL}
            available = {r: [i for i in errors[r] if i not in fresh_sets[r]] for r in EXTERNAL}
            capacity = {r: min(counts[r], len(available[r])) for r in EXTERNAL}
            replay_total = min(64 - fresh_total, fresh_total, sum(capacity.values()))
            replay_total -= replay_total % 2
            candidates.append((fresh_total + replay_total, -fresh_total, counts, indices,
                               available, capacity, replay_total))
            if no_errors or fresh_total + replay_total == 64:
                # With no errors, maximal normal coverage is optimal. With
                # replay, the first full candidate has the most replay allowed.
                break
        require(candidates, 'No exact batch can preserve all external route budgets')
        _, negative_fresh, counts, indices, available, capacity, replay_total = max(candidates, key=lambda v: v[:2])
        fresh_total = -negative_fresh
        replay_counts = dict.fromkeys(EXTERNAL, 0)
        for _ in range(replay_total):
            route = min((r for r in EXTERNAL if replay_counts[r] < capacity[r]),
                        key=lambda r: (replay_counts[r] / counts[r], EXTERNAL.index(r)))
            replay_counts[route] += 1
        slots = []
        for route in EXTERNAL:
            slots += [{'index': i, 'source': 'fresh', 'route': route} for i in indices[route]]
            self.state['cursors'][route] += counts[route]
            pool = available[route]
            cursor = self.state['replay_rotors'][route]
            slots += [{'index': pool[(cursor + offset) % len(pool)], 'source': 'replay', 'route': route}
                      for offset in range(replay_counts[route])]
            self.state['replay_rotors'][route] += replay_counts[route]
        self.state['fresh_groups'] += fresh_total
        slots += self._combo_self_draws(fresh_total + replay_total)
        random.Random(int(self.settings['seed']) + step).shuffle(slots)
        return slots, {'error_pool_sizes': {r: len(errors[r]) for r in EXTERNAL},
                       'feedback_problem': error_problem, 'external_replay_target': .5,
                       'normal_external_queries': fresh_total, 'normal_external_by_route': counts,
                       'replay_queries': replay_total, 'replay_by_route': replay_counts,
                       'self_queries': fresh_total + replay_total,
                       'normal_remaining_by_route': {r: remaining[r] - counts[r] for r in EXTERNAL},
                       'self_strata_target': self.self_mix}, self._fixed_coefficients()

    def _combo_token_replay_plan(self, step):
        """Fit replay inside observed-token quotas, retaining a fixed normal budget.

        Each route keeps at least one normal draw. Replay never exceeds that
        route's normal draws, and fresh/replay are disjoint inside each batch.
        Route pools may wrap independently; only the combined normal budget is
        exhausted. Self matches every external exposure including replay.
        """
        from opd.token_budget import query_shares, quotas
        options = self.settings['token_budget_sampling']
        remaining = self.normal_external_budget - sum(self.state['cursors'].values())
        require(remaining >= 4 and remaining % 2 == 0,
                'Token replay tail cannot retain all external routes')
        shares = query_shares(self.state['token_sampling'], options, self.state['last_committed_step'])
        errors = {r: [] for r in EXTERNAL}
        error_problem = None
        if step > int(self.settings['online_replay'].get('warmup_updates', 10)):
            try:
                errors = self._error_candidates()
            except Exception as exc:
                error_problem = f'{type(exc).__name__}: {exc}'
                print('Online Judge feedback unavailable; use fresh queries: ' + error_problem, flush=True)
        selected = None
        # Largest possible even external batch first. Optional replay can never
        # cause the final normal tail to contain only two slots.
        for external_total in range(64, 3, -2):
            counts = quotas(external_total, shares, int(options.get('min_queries_per_route', 2)))
            replay_counts = {r: counts[r] // 2 for r in EXTERNAL}
            while True:
                fresh_counts = {r: counts[r] - replay_counts[r] for r in EXTERNAL}
                fresh_indices = self._combo_fresh_indices(fresh_counts)
                fresh_sets = {r: set(fresh_indices[r]) for r in EXTERNAL}
                available = {r: [i for i in errors[r] if i not in fresh_sets[r]] for r in EXTERNAL}
                reduced = {r: min(replay_counts[r], len(available[r])) for r in EXTERNAL}
                if reduced == replay_counts:
                    break
                replay_counts = reduced
            # Both normal and replay totals must be even for exact 1:1 and DP4.
            # Converting one optional replay to normal can exclude another error;
            # repeat the finite monotone reduction until parity is also stable.
            while sum(replay_counts.values()) % 2:
                route = max(EXTERNAL, key=lambda r: (replay_counts[r], -EXTERNAL.index(r)))
                replay_counts[route] -= 1
                while True:
                    fresh_counts = {r: counts[r] - replay_counts[r] for r in EXTERNAL}
                    fresh_indices = self._combo_fresh_indices(fresh_counts)
                    fresh_sets = {r: set(fresh_indices[r]) for r in EXTERNAL}
                    available = {r: [i for i in errors[r] if i not in fresh_sets[r]] for r in EXTERNAL}
                    reduced = {r: min(replay_counts[r], len(available[r])) for r in EXTERNAL}
                    if reduced == replay_counts:
                        break
                    replay_counts = reduced
            fresh_counts = {r: counts[r] - replay_counts[r] for r in EXTERNAL}
            fresh_total = sum(fresh_counts.values())
            if fresh_total > remaining or remaining - fresh_total == 2:
                continue
            require(all(1 <= fresh_counts[r] <= self.route_sizes[r]
                        and replay_counts[r] <= fresh_counts[r] for r in EXTERNAL),
                    'Invalid token-budget route normal/replay counts')
            selected = (counts, fresh_counts, fresh_indices, available, replay_counts, fresh_total, external_total)
            break
        require(selected is not None, 'No token-budget batch can preserve the remaining normal budget')
        counts, fresh_counts, fresh_indices, available, replay_counts, fresh_total, external_total = selected
        slots = []
        for route in EXTERNAL:
            slots.extend({'index': i, 'source': 'fresh', 'route': route} for i in fresh_indices[route])
            self.state['cursors'][route] += fresh_counts[route]
            cursor = self.state['replay_rotors'][route]
            pool = available[route]
            slots.extend({'index': pool[(cursor + offset) % len(pool)], 'source': 'replay', 'route': route}
                         for offset in range(replay_counts[route]))
            self.state['replay_rotors'][route] += replay_counts[route]
        self.state['fresh_groups'] += fresh_total
        slots += self._combo_self_draws(external_total)
        self.state['token_sampling']['last_external_quotas'] = dict(counts)
        random.Random(int(self.settings['seed']) + step).shuffle(slots)
        return slots, {'sampling_policy': 'token_budget_replay', 'coverage_mode': 'fixed_external_budget',
                       'error_pool_sizes': {r: len(errors[r]) for r in EXTERNAL},
                       'feedback_problem': error_problem, 'external_replay_target': .5,
                       'target_token_shares': dict(self.state['token_sampling']['target_token_shares']),
                       'query_shares': dict(shares), 'external_quotas': counts,
                       'length_ema': dict(self.state['token_sampling']['length_ema']),
                       'normal_external_queries': fresh_total, 'normal_external_by_route': fresh_counts,
                       'normal_external_remaining': remaining - fresh_total,
                       'replay_queries': sum(replay_counts.values()), 'replay_by_route': replay_counts,
                       'self_queries': external_total, 'self_strata_target': self.self_mix}, self._fixed_coefficients()

    def _fixed_coefficients(self):
        if self.settings.get('loss_aggregation') == 'global_token_mean':
            return None
        coefficients = dict(self.settings.get('route_loss_coefficients') or dict.fromkeys(ROUTES, 1 / 3))
        require(set(coefficients) == set(ROUTES), 'All four teacher loss coefficients must be explicit')
        return coefficients

    def _normal_epoch_plan(self, step):
        """Draw only the next normal candidates; no feedback or replay path."""
        remaining = {r: self.route_budgets[r] - self.state['cursors'][r] for r in EXTERNAL}
        normal_left = sum(remaining.values())
        require(normal_left > 0 and all(remaining.values()), 'Normal epoch route budget exhausted prematurely')
        fresh_total = min(self.external_batch_capacity, normal_left)
        # For other valid pool sizes, leave four external tail slots instead of
        # two, so the tail can contain all three routes and remain DP4-divisible.
        if normal_left - fresh_total == 2:
            fresh_total -= 2
        counts = self._combo_fresh_counts(fresh_total, remaining)
        require(counts is not None, 'Normal batch cannot preserve exact external route coverage')
        indices = self._combo_fresh_indices(counts)
        slots = []
        for route in EXTERNAL:
            slots += [{'index': i, 'source': 'fresh', 'route': route} for i in indices[route]]
            self.state['cursors'][route] += counts[route]
        self.state['fresh_groups'] += fresh_total
        self_count = int(fresh_total / self.external_self_ratio)
        require(self_count * self.external_self_ratio == fresh_total, 'Noninteger tail self exposure')
        slots += self._combo_self_draws(self_count)
        random.Random(int(self.settings['seed']) + step).shuffle(slots)
        return slots, {'sampling_policy': 'normal_epochs', 'normal_external_queries': fresh_total,
                       'normal_external_by_route': counts, 'self_queries': self_count,
                       'normal_remaining_by_route': {r: remaining[r] - counts[r] for r in EXTERNAL},
                       'self_strata_target': self.self_mix}, self._fixed_coefficients()

    def _draw_unique(self, route, count):
        pool = self.pool[route]
        require(count <= len(pool), 'Not enough distinct queries to fill a D3 batch: ' + route)
        cursor = self.state['cursors'][route]
        indices = [pool[(cursor + j) % len(pool)] for j in range(count)]
        self.state['cursors'][route] += count
        require(len(set(indices)) == len(indices), 'Duplicate dynamic query within batch')
        return indices

    def _d3_plan(self, step):
        if self.combo:
            return self._combo_d3_plan(step)
        options = self.settings['dynamic_sampling']
        if self.state['last_committed_step'] >= int(options.get('warmup_updates', 10)):
            committed = self.state['last_committed_step']
            if committed % int(options.get('update_every', 5)) == 0:
                try:
                    ratios, diagnostics = d3_mixture(self.state['history'], options)
                    self.state['mixture'], self.state['mixture_diagnostics'] = ratios, diagnostics
                except Exception as exc:
                    # Keep last valid ratios. Missing metrics fall back uniformly.
                    self.state['mixture_diagnostics'] = {'state': 'retained_last_valid', 'error': str(exc)}
        size = min(self.batch_size, 4720 - self.state['exposures'])
        self_count = size // 4
        remaining_self = 1180 - self.state['self_cursor']
        require(0 < self_count <= remaining_self and size % 4 == 0, 'D3 self/tail budget differs')
        ratios = self.state['mixture']
        counts = largest_remainder(size - self_count, ratios,
                                   float(options.get('ratio_floor', .2)), float(options.get('ratio_cap') or 1.))
        slots = []
        for route in EXTERNAL:
            slots += [{'index': i, 'source': 'dynamic', 'route': route}
                      for i in self._draw_unique(route, counts[route])]
        slots += [{'index': i, 'source': 'fresh', 'route': 'self'}
                  for i in self.original['self'][self.state['self_cursor']:self.state['self_cursor'] + self_count]]
        self.state['self_cursor'] += self_count
        # Preserve exact baseline order during initial equal-mixture phase.
        if step <= int(options.get('warmup_updates', 10)):
            order = self.baseline[(step - 1) * 128:(step - 1) * 128 + size]
            require(Counter(self.entries[i]['teacher_key'] for i in order) == Counter({**counts, 'self': self_count}),
                    'Original warmup order is not balanced')
            sources = {slot['index']: slot for slot in slots}
            require(set(order) == set(sources), 'D3 warmup differs from original inputs')
            slots = [sources[i] for i in order]
        else:
            random.Random(int(self.settings['seed']) + step).shuffle(slots)
        realized = {r: n / (size - self_count) for r, n in counts.items()}
        return slots, {'mixture': dict(ratios), 'realized_mixture': realized,
                       'diagnostics': copy.deepcopy(self.state['mixture_diagnostics']),
                       'external_quotas': counts}, {**realized, 'self': 1 / 3}

    def _combo_d3_plan(self, step):
        options = self.settings['dynamic_sampling']
        committed = self.state['last_committed_step']
        warmup = int(options.get('warmup_updates', math.ceil(.1 * self.maximum_updates)))
        if committed >= warmup and (committed - warmup) % int(options.get('update_every', 5)) == 0:
            ratios, diagnostics = d3_mixture(self.state['history'], options)
            self.state['mixture'], self.state['mixture_diagnostics'] = ratios, diagnostics
        left = self.normal_external_budget - sum(self.state['cursors'].values())
        external = min(self.external_batch_capacity, left)
        if left - external == 2:
            external -= 2
        require(external >= 3, 'Dynamic tail cannot retain all external teacher routes')
        ratios = self.state['mixture']
        counts = largest_remainder(external, ratios, float(options.get('ratio_floor', 1 / 6)),
                                   float(options.get('ratio_cap', .5)))
        slots = []
        for route in EXTERNAL:
            slots += [{'index': index, 'source': 'dynamic', 'route': route}
                      for index in self._draw_unique(route, counts[route])]
        self.state['fresh_groups'] += external
        slots += self._combo_self_draws(external)
        random.Random(int(self.settings['seed']) + step).shuffle(slots)
        realized = {route: counts[route] / external for route in EXTERNAL}
        return slots, {'sampling_policy': 'dynamic_budget', 'mixture': dict(ratios),
                       'realized_mixture': realized, 'diagnostics': copy.deepcopy(self.state['mixture_diagnostics']),
                       'external_quotas': counts, 'normal_external_queries': external,
                       'self_queries': external, 'self_strata_target': self.self_mix}, {**ratios, 'self': 1 / 3}

    def next_plan(self):
        snapshot = copy.deepcopy(self.state)
        previous_plan = self.last_plan
        try:
            return self._next_plan_unchecked()
        except Exception:
            # A failed plan write or invalid selection must not consume cursors.
            self.state, self.last_plan = snapshot, previous_plan
            raise

    def _next_plan_unchecked(self):
        require(self.state['pending_batch'] is None, 'Finish the previous optimizer batch before drawing again')
        if self.state['complete']:
            return None
        step = self.state['next_step']
        path = self.plan_root / f'step_{step:04d}.json'
        before = {key: copy.deepcopy(self.state[key]) for key in self.core_fields}
        if path.is_file():
            plan = read_json(path)
            require(plan['contract'] == self.contract and plan['before'] == before,
                    'Issued online batch differs from restored checkpoint: ' + str(path))
            for key in self.core_fields:
                self.state[key] = copy.deepcopy(plan['after'][key])
            if self.policy == 'd3_composite':
                self.state['mixture'] = copy.deepcopy(plan['diagnostics']['mixture'])
                self.state['mixture_diagnostics'] = copy.deepcopy(plan['diagnostics']['diagnostics'])
        else:
            if self.policy == 'normal_epochs':
                slots, diagnostics, coefficients = self._normal_epoch_plan(step)
            elif self.policy == 'error_replay':
                slots, diagnostics, coefficients = self._replay_plan(step)
            else:
                slots, diagnostics, coefficients = self._d3_plan(step)
            require(slots and len(slots) <= 128 and len(slots) % 4 == 0,
                    'Online batch/tail must divide attention DP4 and four Agent workers')
            identities = [(s['route'], self.entries[s['index']]['query_fingerprint']) if self.combo
                          else self.entries[s['index']]['query_fingerprint'] for s in slots]
            require(len(set(identities)) == len(slots),
                    'Duplicate query in issued online batch')
            for slot in slots:
                route, source = slot['route'], slot['source']
                self.state['exposures_by_route'][route] += 1
                self.state['fresh_by_route'][route] += int(source in ('fresh', 'dynamic'))
                self.state['replay_by_route'][route] += int(source == 'replay')
            self.state['exposures'] += len(slots)
            self.state['next_step'] = step + 1
            self.state['last_plan_step'] = step
            self.state['complete'] = ((sum(self.state['cursors'].values()) == self.normal_external_budget
                                      if self.dynamic or self.token_budget else all(self.state['cursors'][r] == self.route_budgets[r] for r in EXTERNAL))
                                      if self.combo else self.state['fresh_groups'] == 1180
                                      if self.policy == 'error_replay' else self.state['exposures'] == 4720)
            plan = {'schema': 'mopd_combo_normal_batch_v1' if self.normal else
                              'mopd_combo_online_batch_v1' if self.combo else 'mopd_exp5_online_batch_v1',
                    'contract': self.contract, 'step': step,
                    'before': before, 'after': {key: copy.deepcopy(self.state[key]) for key in self.core_fields},
                    'slots': slots, 'coefficients': coefficients, 'diagnostics': diagnostics,
                    'feedback_through_step': None if self.normal else self.state['last_committed_step'],
                    'last': self.state['complete']}
            write_json(path, plan)
        self.state['pending_batch'] = step
        self.last_plan = plan
        return plan

    def commit(self, step, metrics):
        require(self.state['pending_batch'] == step, 'Optimizer commit does not match issued batch')
        committed = copy.deepcopy(self.state)
        if self.token_budget:
            from opd.token_budget import observe
            committed['token_sampling'] = observe(committed['token_sampling'], self.settings['token_budget_sampling'],
                                                  metrics, self.last_plan['slots'])
        if self.policy == 'd3_composite':
            for route in EXTERNAL:
                value = metrics.get(f'opd/teacher/{route}/loss')
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                        'Missing D3 route KL: ' + route)
                committed['history'][route].append(float(value))
        committed['last_committed_step'] = step
        committed['pending_batch'] = None
        write_json(self.run / 'sampling/state.json', committed)
        self.state = committed
        if self.token_budget:
            # Trainer logs this same dictionary after commit, so the controller's
            # targets and actual counters are visible in ordinary step/W&B logs.
            token = committed['token_sampling']
            total_tokens = sum(token['tokens_by_route'].values())
            batch_tokens = sum(metrics[f'opd/teacher/{r}/tokens'] for r in EXTERNAL)
            for route in EXTERNAL:
                prefix = f'opd/token_sampling/{route}/'
                metrics.update({prefix + 'target_token_share': token['target_token_shares'][route],
                                prefix + 'target_query_share': token['query_shares'][route],
                                prefix + 'batch_token_share': metrics[f'opd/teacher/{route}/tokens'] / batch_tokens
                                    if batch_tokens else 0.,
                                prefix + 'cumulative_token_share': token['tokens_by_route'][route] / total_tokens
                                    if total_tokens else 0.,
                                prefix + 'cumulative_tokens': token['tokens_by_route'][route],
                                prefix + 'cumulative_queries': token['queries_by_route'][route],
                                prefix + 'external_quota': token['last_external_quotas'][route]})
                if token['length_ema'][route] is not None:
                    metrics[prefix + 'length_ema'] = token['length_ema'][route]
                for source in ('fresh', 'replay'):
                    metrics[prefix + source + '_queries'] = token['queries_by_source'][source][route]
                    if token['source_token_metrics_available']:
                        metrics[prefix + source + '_tokens'] = token['tokens_by_source'][source][route]

    def outcome(self):
        require(self.state['complete'] and self.state['pending_batch'] is None, 'Online sampler not complete')
        replay = sum(self.state['replay_by_route'].values())
        if self.combo:
            normal = sum(self.state['fresh_by_route'][r] for r in EXTERNAL)
            external = normal + replay
            return {'online_sampling_complete': True, 'steps': self.state['last_committed_step'],
                    'sampling_policy': self.policy,
                    'exposures': self.state['exposures'], 'fresh_exposures': sum(self.state['fresh_by_route'].values()),
                    'fresh_by_route': dict(self.state['fresh_by_route']), 'normal_external_queries': normal,
                    'normal_external_budget': self.normal_external_budget, 'epochs': self.epochs,
                    'normal_external_epochs': self.epochs,
                    'coverage_mode': 'fixed_external_budget' if self.dynamic or self.token_budget else 'complete_external_epochs',
                    'external_self_ratio': self.external_self_ratio,
                    'normal_epoch_coverage_by_route': {r: self.state['cursors'][r] / self.route_sizes[r] for r in EXTERNAL},
                    'replay_exposures': replay, 'replay_by_route': dict(self.state['replay_by_route']),
                    'self_exposures': self.state['self_cursor'], 'self_by_stratum': dict(self.state['self_by_stratum']),
                    'self_repeats_by_stratum': {r: max(0, self.state['self_by_stratum'][r] - len(self.self_pools[r]))
                                               for r in EXTERNAL},
                    'self_strata_target': dict(self.self_mix),
                    'exposures_by_route': dict(self.state['exposures_by_route']),
                    'realized_external_replay_fraction': replay / external,
                    'realized_replay_fraction_by_route': {r: self.state['replay_by_route'][r]
                                                        / self.state['exposures_by_route'][r] for r in EXTERNAL},
                    'progress_fraction': self.progress_fraction(),
                    'token_budget_sampling': copy.deepcopy(self.state.get('token_sampling')),
                    'controller_state_file': str(self.run / 'sampling/state.json')}
        return {'online_sampling_complete': True, 'steps': self.state['last_committed_step'],
                'exposures': self.state['exposures'], 'fresh_exposures': sum(self.state['fresh_by_route'].values()),
                'fresh_by_route': dict(self.state['fresh_by_route']),
                'replay_exposures': replay, 'replay_by_route': dict(self.state['replay_by_route']),
                'self_exposures': self.state['self_cursor'], 'exposures_by_route': self.state['exposures_by_route'],
                'realized_external_replay_fraction': replay / (3540 + replay) if self.policy == 'error_replay' else 0.,
                'controller_state_file': str(self.run / 'sampling/state.json')}


class OnlineDataLoader:
    """Minimal single-process stateful loader, with no speculative prefetch.

    The native trainer uses len/iteration/state_dict/load_state_dict only.
    Selection happens when __next__ is called after the prior optimizer update.
    """
    def __init__(self, dataset, collate_fn, settings):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.scheduler = OnlineScheduler(settings, dataset.pool.entries, dataset.sampling_order)
        self.sampler = self
        self.current_plan = None

    def __len__(self):
        return self.scheduler.maximum_updates

    def __iter__(self):
        return self

    def __next__(self):
        plan = self.scheduler.next_plan()
        if plan is None:
            raise StopIteration
        self.current_plan = plan
        rows = []
        for slot in plan['slots']:
            row = self.dataset[slot['index']]
            row['sampling_source'] = slot['source']
            row['sampling_plan_step'] = plan['step']
            rows.append(row)
        return self.collate_fn(rows)

    def state_dict(self):
        return {'schema': 'mopd_exp5_online_loader_v1', 'scheduler': self.scheduler.state_dict()}

    def load_state_dict(self, state):
        require(state.get('schema') == 'mopd_exp5_online_loader_v1', 'Wrong online dataloader checkpoint')
        self.scheduler.load_state_dict(state['scheduler'])
        self.current_plan = None


def validate_online_resume(settings, entries, baseline_order):
    """Exercise the exact issued-plan/commit/checkpoint protocol before GPU use."""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='opd_exp5_resume_') as temporary:
        local = dict(settings, run_dir=temporary)
        first = OnlineScheduler(local, entries, baseline_order)
        plan = first.next_plan()
        metrics = {f'opd/teacher/{r}/loss': 1. for r in EXTERNAL}
        if first.token_budget:
            for route in EXTERNAL:
                count = sum(slot['route'] == route for slot in plan['slots'])
                metrics[f'opd/teacher/{route}/queries'] = count
                metrics[f'opd/teacher/{route}/tokens'] = count * 100
        first.commit(plan['step'], metrics)
        checkpoint = first.state_dict()
        expected = first.next_plan()
        restored = OnlineScheduler(local, entries, baseline_order)
        restored.load_state_dict(checkpoint)
        require(restored.next_plan() == expected, 'Online checkpoint changed the next issued query batch')
    print('CPU online plan/commit/resume check passed.', flush=True)
