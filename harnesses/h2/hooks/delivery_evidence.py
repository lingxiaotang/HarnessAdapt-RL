"""Observed delivery facts after a successful prepare_patch; never a submission gate."""
import json
import re
HOOK_NAME = 'delivery_evidence'
USE_CONTEXT_BUDGET = True
WORKTREE_REMAINING_TURNS = ()
PROMPT = '<factual delta> This summary is advisory; it does not grade the patch or block submission.'
SMOKE_EVENTS = [
    {'tool_name': 'edit_file', 'arguments': {'path': 'pkg/mod.py'}, 'returncode': 0, 'output': 'EDITED', 'step': 3, 'step_limit': 60},
    {'tool_name': 'prepare_patch', 'arguments': {'files': ['pkg/mod.py']}, 'returncode': 0, 'step': 4, 'step_limit': 60,
     'context_remaining_tokens': 40000,
     'output': '{"ok": true, "files": ["pkg/mod.py"], "related_tests": ["pkg/tests/test_mod.py"], "suggested_check": "python -m pytest -q -x pkg/tests/test_mod.py"}'},
    {'tool_name': 'prepare_patch', 'arguments': {'files': ['pkg/mod.py']}, 'returncode': 0, 'step': 5, 'step_limit': 60, 'output': '{"ok": true}'},
]
SUBMIT = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
RUNNER = re.compile(r'\b(?:pytest|py\.test|nosetests|runtests\.py|manage\.py\s+test)\b|-m\s+unittest\b|unittest\.main\(', re.I)
PYTHON = re.compile(r'\bpython[\d.]*\s+(?:-[a-zA-Z]*c\b|-\s*<<|<<|\S+\.py\b)', re.I)
ASSERT = re.compile(r'\bassert\b|\bassert[A-Z]\w*\(|\braise\s+(?:AssertionError|SystemExit)\b|sys\.exit\((?!0\))')
FAILURE = re.compile(r'^\s*(?:FAILED(?:\s|\()|ERROR:|Traceback \(most recent call last\):|AssertionError\b)|\b[1-9]\d* (?:failed|errors?|exceptions)\b', re.I | re.M)
ABSENT = re.compile(r'No module named|ModuleNotFoundError|command not found|unrecognized arguments|collected 0 items|Ran 0 tests|no tests ran', re.I)
MUTATE = re.compile(r'\bsed\b[^\n]*\s-i\b|\bperl\b[^\n]*\s-p?i\b|\.write_text\(|\.write_bytes\(|open\([^\n]*?["\'](?:w|a|r\+)["\']|\b(?:cp|mv)\s|\bpatch\s+-p|\bgit\s+(?:apply|checkout|restore)\b|>\s*/testbed/|cat\s*>{1,2}\s*(?!\s|/tmp)', re.I)
SOURCE = re.compile(r'(?<![\w/.-])(?:/testbed/)?(?!/)[\w.-]+(?:/[\w.-]+)*\.py\b')
TEST_PATH = re.compile(r'(?:^|/)(?:tests?|testing)/|(?:^|/)test_[^/]*$|_test\.py$|(?:^|/)tests\.py$|(?:^|/)conftest\.py$')
HEREDOC = re.compile(r"cat\s*>\s*(\S+)\s*<<\s*['\"]?(\w+)['\"]?\n(.*?)\n\2\b", re.S)


def _is_scratch(path):
    return str(path or '').startswith(('/tmp', 'tmp/'))


def _shell_edits_source(command):
    if not MUTATE.search(command):
        return False
    if re.search(r'\bgit\s+(?:apply|checkout|restore)\b|\bpatch\s+-p', command):
        return True
    return any(not _is_scratch(path) and not TEST_PATH.search(path) for path in SOURCE.findall(command))


def _check_kind(command, scripts):
    if RUNNER.search(command):
        return 'runner'
    if ASSERT.search(command) or any(has_assert and path in command for path, has_assert in scripts.items()):
        return 'assert'
    return 'print'


def observe(event, state):
    name = event.get('tool_name')
    rc = event.get('returncode')
    step = event.get('step')
    arguments = event.get('arguments') or {}
    output = str(event.get('output') or '')
    scripts = state.setdefault('scripts', {})
    if name in ('edit_file', 'write_file', 'restore_file') and rc == 0:
        path = str(arguments.get('path') or '')
        if _is_scratch(path):
            text = str(arguments.get('content') or arguments.get('new_text') or '')
            scripts[path] = scripts.get(path, False) or bool(ASSERT.search(text))
        else:
            state.update(edit_step=step, checks=[])
        return ''
    command = str(arguments.get('command') or '') if name in ('bash', 'run_check') else ''
    if name == 'bash':
        for target, _, body in HEREDOC.findall(command):
            if _is_scratch(target):
                scripts[target] = bool(ASSERT.search(body))
        if SUBMIT in command:
            return ''
        if _shell_edits_source(command) and 'POLICY_BLOCKED' not in output:
            state.update(edit_step=step, checks=[])
            if not RUNNER.search(command):
                return ''
    if name == 'run_check' or (name == 'bash' and (RUNNER.search(command) or PYTHON.search(command))):
        failed = rc not in (0, None) or bool(FAILURE.search(output)) or bool(ABSENT.search(output))
        if name == 'run_check':
            try:
                meta = json.loads(output.splitlines()[0])
                failed = meta.get('check_exit_code') != 0 or bool(meta.get('failure_signal')) or bool(meta.get('missing_runner_or_zero_tests_signal'))
            except (ValueError, IndexError, AttributeError):
                pass
        if state.get('edit_step') is not None:
            state.setdefault('checks', []).append([step, _check_kind(command, scripts), bool(failed)])
        return ''
    if name != 'prepare_patch' or rc != 0 or state.get('fired', 0) >= 3:
        return ''
    if state.get('fired') and state.get('summarized_edit') == state.get('edit_step'):
        return ''
    try:
        meta = json.loads(output.splitlines()[0])
    except (ValueError, IndexError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    limit = event.get('step_limit')
    remaining = event.get('remaining_turns')
    if type(remaining) is not int and type(limit) is int and type(step) is int:
        remaining = max(0, limit - step)
    tokens = event.get('context_remaining_tokens')
    tight = (type(remaining) is int and type(limit) is int and remaining <= max(5, limit // 12)) or (type(tokens) is int and tokens <= 6144)
    ample = not tight and type(remaining) is int and type(limit) is int and remaining >= max(8, limit // 5)
    run = meta.get('related_test_run') if isinstance(meta.get('related_test_run'), dict) else {}
    status = str(run.get('status') or '')
    command = str(run.get('command') or '')
    failing = [str(x) for x in (run.get('failing') or [])][:5]
    if status in ('passed', 'failed', 'error') and state.get('edit_step') is not None:
        state.setdefault('checks', []).append([step, 'runner', status != 'passed'])
    checks = state.get('checks', []) if state.get('edit_step') is not None else []
    runners = sum(1 for c in checks if c[1] == 'runner')
    asserts = sum(1 for c in checks if c[1] == 'assert')
    prints = sum(1 for c in checks if c[1] == 'print')
    related = [str(p) for p in (meta.get('related_tests') or [])][:5]
    suggestion = str(meta.get('suggested_check') or '')
    large = [str(p) for p in (meta.get('large_deletions') or [])][:3]

    facts = []
    if type(limit) is int and type(step) is int:
        facts.append('Turn %d of %d; %d turns remain.' % (step, limit, remaining))
    if state.get('edit_step') is None:
        facts.append('No tracked source edit was observed.')
    else:
        facts.append('Since the last source edit: %d repository test run(s), %d assertion script(s), %d print-only script(s).' % (runners, asserts, prints))
    if related:
        facts.append('Existing tests referencing the changed code: %s.' % ', '.join(related))
    if status in ('passed', 'failed', 'error'):
        detail = ('failing: %s' % ', '.join(failing)) if failing else str(run.get('summary') or '')[-200:]
        facts.append('prepare_patch ran `%s`: %s%s.' % (command, status, (' (%s)' % detail) if detail else ''))
    elif status == 'timeout':
        facts.append('prepare_patch started `%s` but it exceeded its time limit.' % command)
    if large:
        facts.append('The patch deletes many lines from %s; a shell rewrite can truncate a file. Compare the preview with the intended change and restore unintended deletions.' % ', '.join(large))
    if status in ('failed', 'error'):
        advice = ('Existing tests covering the changed code did not pass after the edit. Inspect the failure (log: %s) before submitting; '
                  'a failure in code you changed is usually a regression, so do not dismiss it as pre-existing without evidence.' % (run.get('log') or 'rerun with run_check'))
    elif checks and checks[-1][2]:
        advice = 'The latest check after the edit failed; decide whether it is an expected reproduction or a regression before submitting.'
    elif runners:
        advice = 'Existing tests ran after the edit. If they and your reproducer cover the original issue, submit with the explicit bash command.'
    elif ample:
        target = ('`%s`' % suggestion) if suggestion else 'the closest existing test file for the changed code'
        advice = 'No existing repository test has run since the edit; print-only scripts do not detect regressions. Run %s with run_check, fix any regression, then submit.' % target
    elif tight:
        advice = 'Budget is tight: submit with the evidence available and retain uncertainty.'
    else:
        advice = 'Few turns remain: run one quick focused existing test only if it fits; otherwise submit.'
    parts = facts + [advice]
    if not state.get('fired'):
        parts.append('Confirm the diff covers the behavior named in the original issue, including explicitly named variants.')
    state['fired'] = state.get('fired', 0) + 1
    state['summarized_edit'] = state.get('edit_step')
    return PROMPT.replace('<factual delta>', ' '.join(parts))
