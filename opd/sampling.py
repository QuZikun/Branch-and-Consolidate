"""Fixed exposure budget; consume unique candidates before repeating any pool."""
from collections import Counter
import hashlib
import json
import math
import random

ROUTES = ('business_reasoning', 'general', 'business_agent', 'self')


def identity(entries):
    return hashlib.sha256(json.dumps([(e['sample_id'], e['teacher_key']) for e in entries],
                                    separators=(',', ':')).encode()).hexdigest()


def allocate_batches(means, draws=4720, batch_size=128):
    if set(means) != set(ROUTES) or not all(math.isfinite(v) and v > 0 for v in means.values()):
        raise ValueError('Need positive archived assistant-token means for all four routes')
    inverse = {r: 1.0 / means[r] for r in ROUTES}
    probabilities = {r: inverse[r] / sum(inverse.values()) for r in ROUTES}
    batches = []
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        if size < len(ROUTES):
            raise ValueError('Tail batch cannot cover every teacher')
        counts = dict.fromkeys(ROUTES, 1)
        for _ in range(size - len(ROUTES)):
            counts[max(ROUTES, key=lambda r: size * probabilities[r] - counts[r])] += 1
        batches.append({'step': len(batches) + 1, 'queries': counts,
                        'estimated_tokens': {r: counts[r] * means[r] for r in ROUTES}})
    return batches


def make_plan(entries, means, batch_size=128, seed=42, draws=4720, self_quotas=None):
    batches = allocate_batches(means, draws, batch_size)
    totals = {r: sum(b['queries'][r] for b in batches) for r in ROUTES}
    pools = {r: [i for i, e in enumerate(entries) if e['teacher_key'] == r] for r in ROUTES}
    if any(not p for p in pools.values()):
        raise ValueError('Every route needs eligible queries')
    sequences = {}
    for route, pool in pools.items():
        pool = list(pool)
        random.Random(f'{seed}:{route}:order').shuffle(pool)
        sequences[route] = [pool[i % len(pool)] for i in range(totals[route])]
    if self_quotas is not None:
        if set(self_quotas) != set(ROUTES[:-1]) or sum(self_quotas.values()) != totals['self']:
            raise ValueError('Self strata must fill the fixed self exposure quota')
        strata = {s: [i for i in pools['self'] if entries[i]['sampling_stratum'] == s] for s in self_quotas}
        for s, pool in strata.items():
            if not pool and self_quotas[s]:
                raise ValueError('Empty self stratum: ' + s)
            random.Random(f'{seed}:self:{s}:order').shuffle(pool)
        used = Counter()
        sequence = []
        for i in range(totals['self']):
            s = max((s for s in strata if used[s] < self_quotas[s]),
                    key=lambda s: (i + 1) * self_quotas[s] / totals['self'] - used[s])
            sequence.append(strata[s][used[s] % len(strata[s])])
            used[s] += 1
        sequences['self'] = sequence
    order, cursors = [], Counter()
    rng = random.Random(seed)
    for batch in batches:
        slots = [route for route in ROUTES for _ in range(batch['queries'][route])]
        rng.shuffle(slots)
        chosen = []
        for route in slots:
            chosen.append(sequences[route][cursors[route]])
            cursors[route] += 1
        if len(chosen) != len(set(entries[i]['query_fingerprint'] for i in chosen)):
            raise ValueError('Pool too small for a duplicate-free batch; reduce its quota explicitly')
        order.extend(chosen)
    unique = {r: len(set(sequences[r])) for r in ROUTES}
    estimated = {r: totals[r] * means[r] for r in ROUTES}
    return {'schema': 'mopd_full_pool_token_draw_v2', 'seed': seed, 'batch_size': batch_size,
            'index_identity': identity(entries), 'order': order, 'draws': len(order),
            'unique_queries': len(set(order)), 'draws_by_route': totals,
            'unique_by_route': unique, 'repeats_by_route': {r: totals[r] - unique[r] for r in ROUTES},
            'self_draws_by_stratum': dict(Counter(entries[i]['sampling_stratum'] for i in sequences['self']))
                                      if self_quotas is not None else {},
            'mean_assistant_tokens': means, 'estimated_tokens_by_route': estimated,
            'estimated_token_shares': {r: estimated[r] / sum(estimated.values()) for r in ROUTES},
            'batches': batches,
            'note': '4720 frozen exposures from full eligible pools. Each route/self stratum consumes its '
                    'unique candidates before cycling; no duplicate query within a batch. Self strata '
                    'retain baseline proportions. Archived lengths estimate token shares; online shares are logged.'}


def read_order(path, entries):
    with open(path) as stream:
        plan = json.load(stream)
    order = plan['order']
    if plan['schema'] == 'mopd_combo_external_seed_v1':
        external = [i for i, entry in enumerate(entries) if entry['teacher_key'] != 'self']
        sizes = dict(Counter(entries[i]['teacher_key'] for i in external))
        route_ids = [(e['teacher_key'], e['query_fingerprint']) for e in entries]
        if (plan.get('index_identity') != identity(entries) or plan.get('batch_size') != 128
                or type(plan.get('epochs')) is not int or plan['epochs'] <= 0
                or plan.get('draws') != len(external)
                or plan.get('route_sizes') != sizes or set(sizes) != set(ROUTES[:-1])
                or order != external or len({e['sample_id'] for e in entries}) != len(entries)
                or len(set(route_ids)) != len(entries)
                or not any(e['teacher_key'] == 'self' for e in entries)):
            raise ValueError('Combination normal external seed/query identity differs')
        return order
    if plan['schema'] in ('mopd_exp5_query_draw_v1', 'mopd_exp5_static_draw_v1'):
        baseline = plan['schema'] == 'mopd_exp5_query_draw_v1'
        indexed = entries[:4720] if baseline else entries
        quotas = (dict.fromkeys(ROUTES, 1180) if baseline else
                  {'business_reasoning': 708, 'general': 1416, 'business_agent': 1416, 'self': 1180})
        if (plan.get('index_identity') != identity(indexed) or plan.get('batch_size') != 128
                or plan.get('draws') != 4720 or len(order) != 4720
                or not all(type(i) is int and 0 <= i < len(indexed) for i in order)
                or len({e['sample_id'] for e in entries}) != len(entries)
                or len({e['query_fingerprint'] for e in entries}) != len(entries)
                or plan.get('draws_by_route') != quotas
                or Counter(entries[i]['teacher_key'] for i in order) != Counter(quotas)):
            raise ValueError('Exp5 frozen query identity/exposure quota differs')
        if baseline and (len(entries) < 4720 or order != list(range(4720))
                         or any(e['teacher_key'] not in ROUTES[:-1] for e in entries[4720:])):
            raise ValueError('Exp5 online seed must be the exact original 4720-query prefix; extras are external only')
        if len(plan.get('batches', [])) != 37:
            raise ValueError('Exp5 seed/static plans require 37 complete batch records')
        for start in range(0, 4720, 128):
            batch = order[start:start + 128]
            observed = dict(Counter(entries[i]['teacher_key'] for i in batch))
            recorded = plan['batches'][start // 128]
            if (len({entries[i]['query_fingerprint'] for i in batch}) != len(batch)
                    or set(observed) != set(ROUTES)
                    or baseline and len(set(observed.values())) != 1
                    or recorded != {'step': start // 128 + 1, 'queries': observed}):
                raise ValueError('Exp5 batch has duplicate queries, missing routes, or changed batch records')
        if not baseline:
            pools = {r: {i for i, e in enumerate(entries) if e['teacher_key'] == r} for r in ROUTES}
            if len(pools['self']) != 1180 or any(not pool for pool in pools.values()):
                raise ValueError('Static exp5 must retain all 1180 original self queries and three eligible external pools')
            seen = {r: set() for r in ROUTES}
            for i in order:
                route = entries[i]['teacher_key']
                if i in seen[route] and seen[route] != pools[route]:
                    raise ValueError('Static exp5 repeated a query before consuming its eligible unique pool')
                seen[route].add(i)
            unique = {r: len(seen[r]) for r in ROUTES}
            if (plan.get('unique_queries') != len(set(order))
                    or plan.get('unique_by_route') != unique
                    or plan.get('repeats_by_route') != {r: quotas[r] - unique[r] for r in ROUTES}):
                raise ValueError('Static exp5 unique/repeat accounting differs')
        elif plan.get('unique_queries') != 4720:
            raise ValueError('Exp5 seed must contain 4720 unique queries')
        return order
    if plan['schema'] == 'mopd_exp3_query_draw_v1':
        batch_size = plan.get('batch_size')
        if (plan.get('index_identity') != identity(entries)
                or batch_size not in (32, 128, 512)
                or len(entries) != 4720 or len(order) != 4720
                or plan.get('draws') != 4720
                or not all(type(i) is int and 0 <= i < 4720 for i in order)
                or set(order) != set(range(4720))
                or len({e['query_fingerprint'] for e in entries}) != 4720
                or Counter(e['teacher_key'] for e in entries)
                   != Counter(dict.fromkeys(ROUTES, 1180))):
            raise ValueError('Exp3 requires the same 4720 unique baseline queries, once per epoch, with BS32/128/512')
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            if {entries[i]['teacher_key'] for i in batch} != set(ROUTES):
                raise ValueError('Every exp3 optimizer batch must contain all four teacher routes')
        return order
    if plan['schema'] not in ('mopd_full_pool_token_draw_v2', 'mopd_exp2_query_draw_v1',
                              'mopd_exp2_delayed_self_v1') or plan['index_identity'] != identity(entries):
        raise ValueError('Frozen sampling plan does not match the data index')
    allowed_draws = (4720,) if plan['schema'] == 'mopd_full_pool_token_draw_v2' else (4720, 7080)
    if len(order) != plan['draws'] or len(order) not in allowed_draws or plan['batch_size'] != 128:
        raise ValueError('Frozen exposure budget must be 4720 or 7080, with BS128')
    if not all(type(i) is int and 0 <= i < len(entries) for i in order):
        raise ValueError('Sampling index out of range')
    delayed_self = plan['schema'] == 'mopd_exp2_delayed_self_v1'
    if delayed_self:
        if (len(entries) != 4720 or len(order) != 4720
                or set(order) != set(range(4720))
                or len({e['query_fingerprint'] for e in entries}) != 4720
                or Counter(e['teacher_key'] for e in entries) != Counter(dict.fromkeys(ROUTES, 1180))
                or plan.get('first_phase_updates') != 18
                or plan.get('unique_queries') != 4720
                or plan.get('draws_by_route') != dict.fromkeys(ROUTES, 1180)):
            raise ValueError('Delayed self must use every original query once: 1180 per route, 18 external-only updates')
        if Counter(entries[i]['teacher_key'] for i in order[:18 * 128]) != Counter(dict.fromkeys(ROUTES[:-1], 768)):
            raise ValueError('Delayed self first 18 updates must contain 768 queries from each external route')
        if len(plan.get('batches', [])) != 37:
            raise ValueError('Delayed self requires a complete 37-update batch manifest')
    for start in range(0, len(order), 128):
        batch = order[start:start + 128]
        effective_routes = set(ROUTES[:-1]) if delayed_self and start < 18 * 128 else set(ROUTES)
        if (len({entries[i]['query_fingerprint'] for i in batch}) != len(batch)
                or {entries[i]['teacher_key'] for i in batch} != effective_routes):
            raise ValueError('Sampling batch has duplicate queries or a missing teacher')
        if delayed_self:
            number = start // 128 + 1
            recorded = plan['batches'][number - 1]
            observed = {r: sum(entries[i]['teacher_key'] == r for i in batch) for r in ROUTES}
            if recorded.get('step') != number or recorded.get('queries') != observed:
                raise ValueError('Delayed self batch manifest differs from its frozen query order')
    return order
