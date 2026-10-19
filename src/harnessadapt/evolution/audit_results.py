"""Independently reconcile metrics with saved production samples and masks."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from .common import read, sha, write


def unpack(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def verify_results(output):
    output = Path(output).resolve()
    protocol = read(output / 'protocol.json')
    summary = read(output / 'summary.json')
    names = protocol['task_ids']
    slots = protocol['rollouts_per_task']
    if not summary['complete'] or summary['infrastructure_errors']:
        raise ValueError(f'Incomplete results cannot be verified: {output}')
    if summary['task_ids'] != names or summary['trajectories'] != len(names) * slots:
        raise ValueError(f'Result task/sample count mismatch: {output}')
    counts = dict(passed=0, gradient=0, submitted=0, effective_tokens=0,
                  hook_errors=0, tool_exceptions=0)
    manifest = {}
    expected_raw_paths = set()
    for group, name in enumerate(names):
        raw_group = []
        for slot in range(slots):
            path = output / 'samples' / f'{group:04d}.{slot}.json.gz'
            expected_raw_paths.add(path)
            raw_group.append(unpack(path))
            manifest[str(path.relative_to(output))] = sha(path)
        equal_rewards = max(s['reward'] for s in raw_group) - min(s['reward'] for s in raw_group) <= 1e-12
        for slot, raw in enumerate(raw_group):
            post_path = output / 'samples' / f'{group:04d}.{slot}.postprocessed.json.gz'
            post = unpack(post_path)
            manifest[str(post_path.relative_to(output))] = sha(post_path)
            for sample in (raw, post):
                if (sample['group_index'], sample['index'], sample['label']) != (group, slot, name):
                    raise ValueError(f'Sample identity mismatch: {post_path}')
                if not sample['metadata'].get('reward_known'):
                    raise ValueError(f'Unknown reward: {post_path}')
                if sample['reward'] not in (0, 1):
                    raise ValueError(f'Nonbinary strict reward: {post_path}')
                if len(sample['loss_mask'] or []) != sample['response_length']:
                    raise ValueError(f'Policy mask length mismatch: {post_path}')
            expected_mask = [0] * raw['response_length'] if equal_rewards else raw['loss_mask']
            if post['loss_mask'] != expected_mask or post['remove_sample'] != raw['remove_sample'] or post['reward'] != raw['reward']:
                raise ValueError(f'Production group mask/reward mismatch: {post_path}')
            if equal_rewards != bool(post['metadata'].get('zero_variance_group_masked')):
                raise ValueError(f'Zero variance marker mismatch: {post_path}')
            token_count = sum(post['loss_mask'] or [])
            eligible = not post['remove_sample'] and token_count > 0
            submitted = bool(post['metadata'].get('submitted'))
            if eligible:
                if not submitted or len(post['tokens']) < post['response_length'] or len(post['rollout_log_probs'] or []) != post['response_length']:
                    raise ValueError(f'Missing submitted-token/log-probability evidence: {post_path}')
                if post['metadata'].get('token_retokenized') or post['metadata'].get('logprobs_missing'):
                    raise ValueError(f'Approximate token evidence cannot enter gradients: {post_path}')
            if post['reward'] == 1 and not submitted:
                raise ValueError(f'Pass without explicit submission: {post_path}')
            trajectory = Path(post['metadata']['trajectory_record_path']).resolve()
            if output not in trajectory.parents:
                raise ValueError(f'Trajectory belongs to another arm: {trajectory}')
            record = unpack(trajectory)
            manifest[str(trajectory.relative_to(output))] = sha(trajectory)
            if (record['instance_id'], record['evolution_group_index'], record['evolution_sample_slot']) != (name, group, slot):
                raise ValueError(f'Trajectory identity mismatch: {trajectory}')
            if record['official_reward'] != post['reward'] or bool(record['official_submitted']) != submitted:
                raise ValueError(f'Strict outcome mismatch: {trajectory}')
            signal = record['training_signal']
            if signal['enters_policy_gradient'] != eligible or signal['trainable_token_count'] != token_count or signal['optimizer_update_executed']:
                raise ValueError(f'Training signal mismatch: {trajectory}')
            counts['passed'] += post['reward'] == 1
            counts['gradient'] += eligible
            counts['submitted'] += submitted
            counts['effective_tokens'] += token_count
            counts['hook_errors'] += sum(record.get('harness_hook_errors', {}).values())
            counts['tool_exceptions'] += sum(message.get('role') == 'tool' and any(code in str(message.get('content', '')) for code in ('"tool_runtime_error"', '"invalid_tool_result"')) for message in record.get('messages', []))
    actual_raw_paths = {p.resolve() for p in (output / 'samples').glob('*.json.gz') if '.postprocessed.' not in p.name}
    if actual_raw_paths != expected_raw_paths:
        raise ValueError(f'Unexpected or missing raw samples: {output}')
    for key, value in counts.items():
        if summary[key] != value:
            raise ValueError(f'Summary {key} disagrees with raw evidence: {output}')
    n = summary['trajectories']
    if summary['pass_rate'] != counts['passed'] / n or summary['gradient_rate'] != counts['gradient'] / n or summary['optimizer_updates']:
        raise ValueError(f'Rate/update flag mismatch: {output}')
    result = dict(verified=True, trajectories=n, **counts, source_files=manifest,
                  protocol_sha256=sha(output / 'protocol.json'), summary_sha256=sha(output / 'summary.json'))
    write(output / 'raw_audit.json', result)
    return result
