"""Route normalization applied once per global optimizer batch, before DP splitting."""
import torch

BASE_ROUTES = ('business_reasoning', 'general', 'business_agent')


def route_token_weights(mask, routes, coefficients=None, require_equal_queries=True,
                        aggregation='route', global_scale=4.0 / 3.0, allow_missing_self=False):
    if mask.ndim != 2 or len(routes) != len(mask):
        raise ValueError('Route/mask shape mismatch')
    if set(routes) not in (set(BASE_ROUTES), set(BASE_ROUTES) | {'self'}):
        raise ValueError('Every balanced batch must contain all three base routes')
    counts = {r: routes.count(r) for r in set(routes)}
    if require_equal_queries and len(set(counts.values())) != 1:
        raise ValueError('Balanced query counts changed: ' + str(counts))
    supplied_coefficients = coefficients
    coefficients = coefficients or {r: 1.0 / 3.0 for r in counts}
    missing_self_only = (allow_missing_self and set(counts) == set(BASE_ROUTES)
                         and set(coefficients) == set(BASE_ROUTES) | {'self'})
    if set(coefficients) != set(counts) and not missing_self_only:
        raise ValueError('Route coefficient keys differ from effective routes')
    import math
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in coefficients.values()):
        raise ValueError('Route coefficients must be finite and positive')
    lengths = mask.sum(-1)
    if aggregation == 'global_token_mean':
        if supplied_coefficients is not None or not math.isfinite(global_scale) or global_scale <= 0:
            raise ValueError('Global token aggregation needs one positive scale, not route coefficients')
        if not bool((lengths > 0).all()):
            raise ValueError('Empty assistant mask')
        return torch.ones(len(routes), dtype=torch.float32, device=mask.device) * (global_scale / lengths.sum())
    if aggregation not in ('route', 'route_query_mean'):
        raise ValueError('Unknown loss aggregation: ' + aggregation)
    if aggregation == 'route_query_mean' and not bool((lengths > 0).all()):
        raise ValueError('Empty assistant mask')
    weights = torch.empty(len(routes), dtype=torch.float32, device=mask.device)
    for route in counts:
        chosen = torch.tensor([r == route for r in routes], device=mask.device)
        tokens = lengths[chosen].sum()
        if tokens <= 0:
            raise ValueError('Empty route: ' + route)
        if aggregation == 'route_query_mean':
            # Each effective query contributes its own assistant-token mean.
            # Freeze both denominators before any DP/microbatch partitioning.
            weights[chosen] = float(coefficients[route]) / (counts[route] * lengths[chosen])
        else:
            weights[chosen] = float(coefficients[route]) / tokens
    return weights


def weighted_route_loss(values, mask, weights, dp_size, max_clamp):
    if values.shape != mask.shape or len(weights) != len(values):
        raise ValueError('Lost route weights during microbatch splitting')
    values = values.float()
    if max_clamp is not None:
        values = values.clamp(-max_clamp, max_clamp)
    # Native Megatron sums microbatches and averages DP gradients. Denominators
    # came from the whole batch, so do not renormalize inside a microbatch.
    return (torch.where(mask.bool(), values, 0.0) * weights.reshape(-1, 1)).sum() * dp_size
