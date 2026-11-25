"""Checkpoint spacing by committed normal-data coverage, independent of replay."""
import copy
import math

from opd.common import require


class CoverageCheckpointSchedule:
    """Coalesce crossed thresholds into one checkpoint at an optimizer boundary.

    The last threshold is reserved for the actual final update, even if normal
    external coverage has finished while self/replay work is still draining.
    Explicit STOP snapshots can be earlier and do not consume future thresholds.
    """

    SCHEMA = 'mopd_coverage_checkpoint_schedule_v1'

    def __init__(self, settings, state=None):
        self.count = settings.get('count', 20)
        self.start_fraction = float(settings.get('start_fraction', .25))
        self.progress_basis = settings.get('progress_basis', 'normal_external_exposures')
        require(type(self.count) is int and self.count > 0, 'Checkpoint count must be a positive integer')
        require(math.isfinite(self.start_fraction) and 0 <= self.start_fraction < 1,
                'Checkpoint start_fraction must be in [0, 1)')
        require(self.progress_basis == 'normal_external_exposures', 'Unsupported checkpoint progress basis')
        supplied = settings.get('milestone_fractions', [])
        require(isinstance(supplied, (list, tuple)), 'Checkpoint milestone_fractions must be a list')
        self.milestone_fractions = sorted(float(value) for value in supplied)
        require(len(set(self.milestone_fractions)) == len(self.milestone_fractions)
                and all(math.isfinite(value) and self.start_fraction < value < 1.
                        for value in self.milestone_fractions),
                'Checkpoint milestones must be distinct fractions strictly between start_fraction and 1')
        require(len(self.milestone_fractions) <= max(0, self.count - 2),
                'Not enough internal checkpoint thresholds for the requested milestones')
        self.thresholds = ([1.] if self.count == 1 else
                           [self.start_fraction + (1. - self.start_fraction) * i / (self.count - 1)
                            for i in range(self.count)])
        available = set(range(1, self.count - 1))
        for milestone in self.milestone_fractions:
            index = min(available, key=lambda i: (abs(self.thresholds[i] - milestone), i))
            self.thresholds[index] = milestone
            available.remove(index)
        self.thresholds.sort()
        self.events = []
        if state is not None:
            require(state.get('schema') == self.SCHEMA, 'Unknown checkpoint schedule snapshot')
            require(state.get('settings') == self.settings, 'Checkpoint schedule changed on resume')
            for event in state.get('events', []):
                self.record_saved(event)

    @property
    def settings(self):
        settings = {'count': self.count, 'start_fraction': self.start_fraction,
                    'progress_basis': self.progress_basis}
        if self.milestone_fractions:
            settings['milestone_fractions'] = list(self.milestone_fractions)
        return settings

    def state_dict(self):
        return {'schema': self.SCHEMA, 'settings': self.settings, 'events': copy.deepcopy(self.events)}

    def decide(self, step, progress, final=False, force=False):
        require(type(step) is int and step > 0, 'Checkpoint step must be a positive committed update')
        progress = float(progress)
        require(math.isfinite(progress) and 0 <= progress <= 1., 'Checkpoint progress must be in [0, 1]')
        require(not final or progress >= 1. - 1e-12, 'Final checkpoint precedes normal-data completion')
        if self.events:
            require(step >= self.events[-1]['step'], 'Checkpoint optimizer step went backwards')
            require(progress + 1e-12 >= self.events[-1]['progress_fraction'], 'Checkpoint coverage went backwards')
            if step == self.events[-1]['step']:
                return None
        covered = {i for event in self.events for i in event['covered_thresholds']}
        due = [i for i, threshold in enumerate(self.thresholds)
               if i not in covered and progress + 1e-12 >= threshold and (i < self.count - 1 or final)]
        if not due and not force and not final:
            return None
        return {'step': step, 'progress_fraction': progress,
                'reason': 'final' if final else 'scheduled' if due else 'stop_recovery',
                'covered_thresholds': due, 'forced': bool(force)}

    def record_saved(self, event):
        require(event.get('reason') in ('scheduled', 'final', 'stop_recovery'), 'Invalid checkpoint reason')
        expected = self.decide(event['step'], event['progress_fraction'],
                               final=event['reason'] == 'final', force=event.get('forced', False))
        require(expected == event, 'Checkpoint snapshot has duplicate or inconsistent coverage thresholds')
        self.events.append(copy.deepcopy(event))

    def state_after_save(self, event):
        """Stage this in the checkpoint before native writes its latest pointer.

        It is used only after native restores a valid model plus dataloader from
        that exact checkpoint. A failed save never advances the live schedule.
        """
        candidate = CoverageCheckpointSchedule(self.settings, self.state_dict())
        candidate.record_saved(event)
        return candidate.state_dict()
