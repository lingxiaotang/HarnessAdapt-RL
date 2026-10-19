"""Short advisory feedback; neither blocks submission nor claims correctness."""
import json
import re
HOOK_NAME = 'validation_feedback'
WORKTREE_REMAINING_TURNS = ()
PROMPT = '<factual delta> Interpret evidence rather than the shell status alone. Do not change tests to hide a failure.'
SMOKE_EVENTS = [{'tool_name':'run_check','returncode':1,'output':'{"check_exit_code":1,"failure_signal":true}','step':1}]
# Only recognize a direct runner command, not source text mentioning pytest.
RUNNER = re.compile(r'(?:^|&&|;|\n)\s*(?:(?:python[\d.]*|python)\s+(?:-m\s+(?:pytest|unittest)\b|(?:\S*/)?(?:runtests|test[^/\s]*|repro[^/\s]*|check[^/\s]*)\.py\b)|pytest\b|nosetests\b)', re.I)
FAILURE = re.compile(r'^\s*(?:FAILED(?:\s|\()|ERROR:|Traceback \(most recent call last\):|AssertionError\b)|\b[1-9]\d* (?:failed|errors?|exceptions)\b', re.I|re.M)
ABSENT = re.compile(r'No module named|ModuleNotFoundError|command not found|unrecognized arguments|collected 0 items|Ran 0 tests|no tests ran', re.I)

def observe(event,state):
    name=event.get('tool_name');rc=event.get('returncode');output=str(event.get('output') or '')
    if name in ('edit_file','write_file') and rc==0:
        state['edited_since_check']=True
        return ''
    if name=='prepare_patch' and rc!=0 and not state.get('prepare_failure_warning'):
        state['prepare_failure_warning']=True
        return PROMPT.replace('<factual delta>',
            'No new submission artifact was prepared. Read the concrete error and correct the selected source paths or preparation problem, then call prepare_patch again. '
            'Do not hand-write a diff, copy an entire source file to patch.txt, or print the submission marker with a missing-file fallback. '
            'Source changes alone do not imply that patch.txt exists or is current. Keep your worktree edits; a preparation error is not evidence that the repair is wrong.')
    meta=None
    if name=='run_check':
        try:meta=json.loads(output.splitlines()[0])
        except (ValueError,IndexError):return ''
        if not isinstance(meta,dict) or type(meta.get('check_exit_code')) is not int:return ''
    elif name=='bash':
        command=str((event.get('arguments') or {}).get('command') or '')
        # Heredoc bodies can contain code/quoted runner names: do not infer checks there.
        if '<<' not in command and RUNNER.search(command) and type(rc) is int:
            meta={'check_exit_code':rc,'failure_signal':bool(FAILURE.search(output)),
                  'missing_runner_or_zero_tests_signal':bool(ABSENT.search(output))}
    if meta is not None:
        failed=meta['check_exit_code']!=0 or bool(meta.get('failure_signal'))
        absent=bool(meta.get('missing_runner_or_zero_tests_signal'))
        state.update(seen_check=True,edited_since_check=False,last_check_failed=failed or absent)
        if not(failed or absent) or state.get('warnings',0)>=2:return ''
        state['warnings']=state.get('warnings',0)+1
        if absent:
            fact='The check shows a runner/setup problem or zero tests. Use repository instructions to correct one invocation; if unavailable, use a minimal assertion-based /tmp reproducer instead of repeated setup guesses.'
        elif meta.get('timeout_status'):
            fact='The check returned a timeout-like exit code. Choose a smaller test target and inspect its saved log; do not repeat a broad suite unchanged.'
        else:
            fact='The check returned a failure status or failure text. Determine whether this is an expected reproduction, a real regression, or an invocation problem. Inspect the specific mismatch before editing or rerunning.'
        return PROMPT.replace('<factual delta>',fact)
    if name=='prepare_patch' and rc==0 and not state.get('preparation_warning'):
        state['preparation_warning']=True
        if state.get('last_check_failed'):
            fact='Patch preparation succeeded, but the latest recognized check still reported failure. If it was a pre-fix reproducer, rerun it; if it was a regression, inspect it before delivery. Preparation checks structure, not behavior.'
        elif not state.get('seen_check') or state.get('edited_since_check'):
            fact='Patch preparation succeeded. No recognized check after the latest tracked edit was observed. If already checked through another command, use that evidence; otherwise run one focused assertion-based check if budget allows. This is advisory, not a submission gate.'
        else:return ''
        return PROMPT.replace('<factual delta>',fact)
    return ''
