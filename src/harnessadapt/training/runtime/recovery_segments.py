"""Identity and aggregation of training segments belonging to one rollout."""
from collections import OrderedDict
import copy
import os


def enabled():
    return os.environ.get('R2E_PRESERVE_RECOVERY_SEGMENTS', '0').lower() in {'1', 'true', 'yes'}


def mark_segments(samples, base_sample, session_id, reward):
    rid = base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index
    if rid is None or base_sample.group_index is None:
        raise ValueError('Recovery segments require stable rollout_id and group_index')
    for index, sample in enumerate(samples):
        sample.rollout_id = rid
        sample.group_index = base_sample.group_index
        sample.session_id = session_id
        sample.reward = reward
        sample.metadata = {**(sample.metadata or {}), 'recovery_segments': True,
                           'segment_index': index, 'segment_count': len(samples)}
    return samples if len(samples) > 1 else samples[0]


def rollout_groups(samples):
    """Group only explicitly marked siblings; reject incomplete/colliding rows.

    Legacy unmarked samples retain their original one-row-per-rollout behavior.
    The output consists of lists of input row indices, in encounter order.
    """
    groups = OrderedDict()
    for index, sample in enumerate(samples):
        md = sample.metadata or {}
        if md.get('recovery_segments'):
            if sample.rollout_id is None or sample.group_index is None or not sample.session_id:
                raise ValueError('Missing recovery rollout identity')
            key = ('rollout', sample.rollout_id)
        else:
            key = ('legacy', index)
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        first = samples[indices[0]]
        md = first.metadata or {}
        if not md.get('recovery_segments'):
            continue
        count = md['segment_count']
        if len(indices) != count or sorted(samples[i].metadata['segment_index'] for i in indices) != list(range(count)):
            raise ValueError('Missing or duplicate recovery segment')
        for i in indices:
            sample = samples[i]
            if (sample.group_index, sample.session_id, sample.label, sample.metadata['segment_count']) != (
                    first.group_index, first.session_id, first.label, count):
                raise ValueError('Recovery segments disagree on rollout identity')
            if bool(sample.remove_sample) != bool(first.remove_sample):
                raise ValueError('Recovery siblings must share submission eligibility')
    return list(groups.values())


def metric_views(samples):
    """Reporting-only rollout views; never feed these concatenations to a model.

    The actual training samples remain separate causal sequences. Token arrays
    here serve only aggregate size/mask statistics in monitoring.py.
    """
    result = []
    for indices in rollout_groups(samples):
        if len(indices) == 1:
            result.append(samples[indices[0]])
            continue
        parts = [samples[i] for i in indices]
        view = copy.copy(parts[0])
        view.tokens = [t for part in parts for t in (part.tokens or [])]
        view.response_length = sum(part.response_length for part in parts)
        view.loss_mask = [m for part in parts for m in (part.loss_mask or [])]
        view.rollout_log_probs = [p for part in parts for p in (part.rollout_log_probs or [])]
        view.metadata = {**(view.metadata or {}), 'reporting_only_rollout_view': True}
        result.append(view)
    return result
