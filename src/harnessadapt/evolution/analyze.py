"""Recomputable solver-trajectory evidence; never uses gold patches or hidden tests."""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import re

from .common import read, sha, write


def analyze(output):
    output = Path(output)
    paths = sorted((output / 'trajectories/train').glob('*.json.gz'))
    # Completed evidence is immutable while the source trajectories are
    # unchanged. A controller restart must not invalidate an already queued
    # proposal merely by changing JSON key order in an equivalent analysis.
    if (output / 'analysis.json').exists() and (output / 'summary.json').exists():
        summary = read(output / 'summary.json')
        cached = read(output / 'analysis.json')
        references = {str(Path(item['path']).resolve()): item['sha256']
                      for group in cached.get('per_task', {}).values() for item in group}
        if (summary.get('complete') and cached.get('trajectory_count') == summary['trajectories']
                and 'production_masked_reasons' in cached
                and references == {str(p.resolve()): sha(p) for p in paths}):
            return cached
    totals = Counter()
    exits = Counter()
    calls = Counter()
    examples = {}
    per_task = {}
    masked_reasons = Counter()
    forks = 0
    for sample_path in (output / 'samples').glob('*.json.gz'):
        if '.postprocessed.' in sample_path.name:
            continue
        with gzip.open(sample_path, 'rt') as stream:
            sample = json.load(stream)
        metadata = sample.get('metadata') or {}
        if sample.get('remove_sample'):
            masked_reasons[str(metadata.get('masked_reason') or 'unspecified')] += 1
        forks += int(metadata.get('adapter_sample_count') or 0) > 1
    for path in paths:
        with gzip.open(path, 'rt') as stream:
            row = json.load(stream)
        outcome = row['outcome']
        submitted = bool(row['official_submitted'])
        success = row['official_reward'] == 1
        flags = set()
        messages = row.get('messages', [])
        actions = []
        seen = Counter()
        observations = []
        for index, message in enumerate(messages):
            if message.get('role') == 'assistant':
                for action in (message.get('extra') or {}).get('actions', []):
                    name = action.get('tool_name') or 'bash'
                    calls[name] += 1
                    command = str(action.get('command') or '')
                    actions.append((name, command))
                    normalized = {k: v for k, v in action.items() if k != 'tool_call_id'}
                    seen[(name, json.dumps(normalized, sort_keys=True))] += 1
            if message.get('role') == 'tool':
                body = str(message.get('content') or '')
                observations.append((index, len(body), body[:1200]))
                if len(body) > 4500:
                    flags.add('large_observation')
                if 'SKYRL_GIT_POLICY_BLOCKED' in body:
                    flags.add('git_policy_blocked')
                if re.search(r'No module named|command not found|collected 0 items|no tests ran', body):
                    flags.add('setup_or_zero_test')
                if re.search(r'FAILED|AssertionError|Traceback', body):
                    flags.add('observed_failure')
        if max(seen.values(), default=0) >= 3:
            flags.add('repeated_identical_action')
        if outcome.get('context_budget_exit'):
            flags.add('context_exhaustion')
        if int(outcome.get('n_calls') or 0) >= 60:
            flags.add('turn_exhaustion')
        if not submitted:
            flags.add('unsubmitted')
        if row.get('worktree_patch') and not submitted:
            flags.add('edited_unsubmitted')
        if submitted and not success:
            flags.add('submitted_failed')
        if outcome.get('training_patch_apply_failed'):
            flags.add('patch_apply_failed')
        if not any(name == 'run_check' or re.search(r'pytest|unittest|runtests|repro|assert', cmd)
                   for name, cmd in actions):
            flags.add('no_recognized_check')
        totals.update(sorted(flags))
        exits[str(outcome.get('exit_status') or 'unknown')] += 1
        item = dict(path=str(path), sha256=sha(path), task=row['instance_id'],
            slot=row.get('evolution_sample_slot'), passed=success, submitted=submitted,
            turns=outcome.get('n_calls'), flags=sorted(flags),
            final_actions=actions[-8:], final_observations=observations[-3:])
        per_task.setdefault(row['instance_id'], []).append(item)
        for flag in sorted(flags):
            values = examples.setdefault(flag, [])
            if len(values) < 6 and not success:
                values.append(item)
    findings = dict(trajectory_count=sum(len(g) for g in per_task.values()),
        issue_trajectory_counts=dict(totals), exit_statuses=dict(exits), tool_calls=dict(calls),
        production_masked_reasons=dict(masked_reasons), adapter_fork_trajectories=forks,
        examples=examples, per_task=per_task,
        limitations='Command/check classification is a heuristic; raw messages are authoritative. '
                     'This analysis never reads reference patches or grader hidden-test results.')
    write(output / 'analysis.json', findings)
    body = ['# Solver trajectory analysis', '',
            'Counts identify candidate mechanisms; they do not prove causality.', '',
            '| Observed issue | Trajectories |', '|---|---:|']
    body += [f'| {key} | {value} |' for key, value in totals.most_common()]
    body += ['', 'Raw examples and SHA256 references: `analysis.json`.']
    (output / 'analysis.md').write_text('\n'.join(body) + '\n')
    return findings
