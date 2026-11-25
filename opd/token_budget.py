"""Observed-token query quotas; no extra generation or forward pass."""
import copy
import math

from opd.common import require

EXTERNAL = ('business_reasoning', 'general', 'business_agent')


def validate(options):
    require(options.get('enabled') is True, 'Token-budget sampling must be enabled explicitly')
    for name, default in (('warmup_updates', 17), ('update_every', 5), ('min_queries_per_route', 2)):
        value = options.get(name, default)
        require(type(value) is int and value >= (0 if name == 'warmup_updates' else 1),
                'Invalid token-budget ' + name)
    require(options.get('min_queries_per_route', 2) >= 2,
            'Token quotas need at least one normal and one possible replay query per route')
    alpha = options.get('ema_alpha', .2)
    require(type(alpha) in (int, float) and math.isfinite(alpha) and 0 < alpha <= 1,
            'Invalid token-budget EMA alpha')
    shares = options.get('target_token_shares', dict.fromkeys(EXTERNAL, 1 / 3))
    require(set(shares) == set(EXTERNAL)
            and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in shares.values())
            and math.isclose(sum(shares.values()), 1., abs_tol=1e-9),
            'Token target shares must form a positive three-route simplex')
    return shares


def initial_state(options):
    shares = validate(options)
    return {'length_ema': dict.fromkeys(EXTERNAL, None),
            'observed_updates': dict.fromkeys(EXTERNAL, 0),
            'tokens_by_route': dict.fromkeys(EXTERNAL, 0.),
            'queries_by_route': dict.fromkeys(EXTERNAL, 0),
            'queries_by_source': {source: dict.fromkeys(EXTERNAL, 0) for source in ('fresh', 'replay')},
            'tokens_by_source': {source: dict.fromkeys(EXTERNAL, 0.) for source in ('fresh', 'replay')},
            'source_token_metrics_available': True,
            'target_token_shares': dict(shares),
            'query_shares': dict.fromkeys(EXTERNAL, 1 / 3),
            'last_quota_update': 0, 'last_external_quotas': dict.fromkeys(EXTERNAL, 0)}


def query_shares(state, options, committed_step):
    """Freeze an inverse-length mixture only at configured batch boundaries."""
    warmup = options.get('warmup_updates', 17)
    if committed_step < warmup or (committed_step - warmup) % options.get('update_every', 5):
        return dict(state['query_shares'])
    lengths = state['length_ema']
    if not all(value is not None and value > 0 for value in lengths.values()):
        return dict(state['query_shares'])
    weights = {route: state['target_token_shares'][route] / lengths[route] for route in EXTERNAL}
    total = sum(weights.values())
    state['query_shares'] = {route: weights[route] / total for route in EXTERNAL}
    state['last_quota_update'] = committed_step
    return dict(state['query_shares'])


def quotas(total, shares, minimum=2):
    """Reserve route presence, then apportion the remaining slots fairly."""
    require(type(total) is int and total >= 3, 'Token batch cannot retain three routes')
    minimum = min(minimum, total // 3)
    require(minimum >= 1, 'Token tail lost an external route')
    counts = dict.fromkeys(EXTERNAL, minimum)
    for _ in range(total - 3 * minimum):
        route = max(EXTERNAL, key=lambda r: (total * shares[r] - counts[r], -EXTERNAL.index(r)))
        counts[route] += 1
    return counts


def observe(state, options, metrics, slots):
    """Use actual assistant-token/query counters, including replay, excluding self."""
    result = copy.deepcopy(state)
    alpha = float(options.get('ema_alpha', .2))
    for route in EXTERNAL:
        route_slots = [slot for slot in slots if slot['route'] == route]
        queries = metrics.get(f'opd/teacher/{route}/queries')
        tokens = metrics.get(f'opd/teacher/{route}/tokens')
        require(type(queries) in (int, float) and queries == len(route_slots) and queries > 0,
                'Missing/inconsistent token-budget route query counter: ' + route)
        require(type(tokens) in (int, float) and math.isfinite(tokens) and tokens >= 0,
                'Missing/invalid token-budget assistant-token counter: ' + route)
        if tokens > 0:
            length = tokens / queries
            previous = result['length_ema'][route]
            result['length_ema'][route] = length if previous is None else alpha * length + (1 - alpha) * previous
            result['observed_updates'][route] += 1
        result['tokens_by_route'][route] += tokens
        result['queries_by_route'][route] += len(route_slots)
        source_tokens = {}
        for source in ('fresh', 'replay'):
            count = sum(slot['source'] == source for slot in route_slots)
            result['queries_by_source'][source][route] += count
            value = metrics.get(f'opd/sampling/{route}/{source}_tokens')
            if value is None and count == 0:
                value = 0.
            if value is None:
                result['source_token_metrics_available'] = False
            else:
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                        'Invalid token-budget source assistant-token counter')
                source_tokens[source] = value
                result['tokens_by_source'][source][route] += value
        if len(source_tokens) == 2:
            require(math.isclose(sum(source_tokens.values()), tokens, abs_tol=1e-6, rel_tol=1e-8),
                    'Token source counters differ from teacher total')
    return result
