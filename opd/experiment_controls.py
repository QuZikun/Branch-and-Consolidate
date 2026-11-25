"""Pure CPU controls for aggregation and exposure-based negative feedback."""
import math

from opd.common import require

EXTERNAL = ('business_reasoning', 'general', 'business_agent')
AGGREGATIONS = ('route', 'route_query_mean', 'global_token_mean')


def batch_coefficients(settings, sampling_plan=None):
    aggregation = settings.get('loss_aggregation', 'route')
    require(aggregation in AGGREGATIONS, 'Unknown loss aggregation: ' + str(aggregation))
    if aggregation == 'global_token_mean':
        require(settings.get('route_loss_coefficients') is None,
                'Global token mean must not add separate teacher loss coefficients')
        scale = float(settings.get('global_loss_scale', 4.0 / 3.0))
        require(math.isfinite(scale) and scale > 0, 'Invalid global token mean scale')
        # Sampling changes the token population; do not multiply its domain
        # proportions into the objective a second time.
        return None
    return (sampling_plan['coefficients'] if sampling_plan is not None
            else settings.get('route_loss_coefficients'))


def normal_progress_after_batch(plan, normal_external_budget):
    require(isinstance(plan, dict) and type(normal_external_budget) is int
            and normal_external_budget > 0, 'Exposure schedule requires an online batch plan')
    counts = plan['after']['fresh_by_route']
    require(all(type(counts.get(route)) is int and counts[route] >= 0 for route in EXTERNAL),
            'Invalid normal external exposure counters')
    normal = sum(counts[route] for route in EXTERNAL)
    require(normal <= normal_external_budget, 'Normal exposure schedule exceeded its budget')
    return normal / normal_external_budget


def negative_feedback_alpha(options, *, normal_progress=None, step=None):
    """alpha=0 preserves RKL; alpha=.5 halves the negative branch's weight."""
    require(options.get('enabled') is True, 'Negative feedback schedule is disabled')
    end_alpha = float(options['alpha_end'])
    require(math.isfinite(end_alpha) and 0 <= end_alpha <= 1
            and list(options.get('exclude_routes', [])) == ['self'],
            'Negative feedback must retain the fixed self branch')
    if options.get('progress_basis') == 'normal_external_exposures':
        begin, end = float(options['start_fraction']), float(options.get('end_fraction', 1.))
        require(math.isfinite(begin) and math.isfinite(end) and 0 <= begin < end <= 1,
                'Invalid negative feedback progress interval')
        require(type(normal_progress) in (int, float) and math.isfinite(normal_progress)
                and 0 <= normal_progress <= 1, 'Missing committed-batch exposure progress')
        fraction = min(1., max(0., (normal_progress - begin) / (end - begin)))
    else:
        # Retain the previously verified fixed-step adapter for older callers.
        begin, end = int(options['begin_step']), int(options['end_step'])
        require(begin == 19 and end == 37 and end_alpha == .5
                and type(step) is int and step > 0,
                'Legacy negative feedback schedule differs from the frozen exp5 contract')
        fraction = min(1., max(0., (step - begin + 1) / (end - begin + 1)))
    return end_alpha * fraction
