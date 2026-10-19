"""Bound check wall time, preserve failures, and retain full diagnostics off-context."""
import json
import shlex
TOOL_SCHEMA = {'type':'function','function':{
 'name':'run_check',
 'description':'Run one focused test or assertion-based /tmp reproducer in /testbed. Preserve pipeline failures; save the full output to a unique /tmp log and return a compact diagnostic view. Default timeout 45 seconds; narrow a check that times out. Do not pipe through head/tail/grep or hide failures with || true. A zero exit is not proof of correctness.',
 'parameters':{'type':'object','properties':{
  'command':{'type':'string','description':'The test/reproducer command; use the supported repository runner and real assertions.'},
  'timeout_seconds':{'type':'integer','minimum':1,'maximum':45,'description':'Wall time limit in seconds, default 45, capped below the environment command timeout.'}},
 'required':['command'],'additionalProperties':False}}}
SMOKE_ARGUMENTS = {'command':"python3 -c 'assert 2 + 2 == 4; print(\"assertions passed\")'"}
_REPORT = r'''
import sys,json,re,collections
path,rc=sys.argv[1],int(sys.argv[2])
head=[];tail=collections.deque(maxlen=6);signals=[];chars=0;failed=False;absent=False
failure=re.compile(r'^\s*(?:FAILED(?:\s|\()|ERROR:|Traceback \(most recent call last\):|AssertionError\b)|\b[1-9]\d* (?:failed|errors?|exceptions)\b',re.I)
missing=re.compile(r'No module named|ModuleNotFoundError|command not found|unrecognized arguments|collected 0 items|Ran 0 tests|no tests ran',re.I)
pattern=re.compile(r'FAILED|ERROR|Traceback|AssertionError|No module named|command not found|unrecognized arguments|collected \d+|Ran \d+ tests?|\b\d+ (?:failed|passed|errors?|skipped)\b|^OK\b',re.I)
with open(path,errors='replace') as f:
 for line in f:
  chars+=len(line);short=line.rstrip()[:300]
  if len(head)<2:head.append(short)
  tail.append(short)
  failed=failed or bool(failure.search(line));absent=absent or bool(missing.search(line))
  if pattern.search(line):
   signals.append(short)
   if len(signals)>5:signals.pop(0)
print(json.dumps({'check_exit_code':rc,'log_path':path,'output_chars':chars,'failure_signal':failed,'missing_runner_or_zero_tests_signal':absent,'timeout_status':rc in (124,137),'scope':'Text signals are advisory; inspect actual assertions/test counts. Exit 0 alone is not proof.'}))
print('DIAGNOSTIC LINES:');print('\n'.join(signals)[:1000])
print('OUTPUT HEAD:');print('\n'.join(head)[:400])
print('OUTPUT TAIL:');print('\n'.join(tail)[-1000:])
print('Full output is in log_path. Read only a relevant window if necessary. No tests or syntax-only checks do not validate behavior.')
'''
def execute(arguments,environment):
    error=lambda s:{'output':s,'returncode':2,'exception_info':''}
    if not isinstance(arguments,dict) or set(arguments)-{'command','timeout_seconds'} or 'command' not in arguments:
        return error('run_check requires command and optionally timeout_seconds.')
    command=arguments['command'];seconds=arguments.get('timeout_seconds',45)
    if not isinstance(command,str) or not command.strip() or len(command)>12000 or '\x00' in command:
        return error('Use a nonempty command <=12000 characters, without NUL.')
    if type(seconds)is not int or not 1<=seconds<=45:
        return error('timeout_seconds must be an integer from 1 to 45.')
    # Keep the command as a visible shell function body for the native policy.
    # timeout runs the exported function in its own shell/process group.
    shell=('cd /testbed || exit 2\nset +e\nharness_check_log=$(mktemp /tmp/harness-check.XXXXXX) || exit 2\n'
           +'harness_check_body() (\nset -o pipefail;\n'+command+'\n)\nexport -f harness_check_body\n'
           +'timeout --signal=TERM --kill-after=2s '+str(seconds)+'s bash -c harness_check_body'
           +' >"$harness_check_log" 2>&1\nharness_check_rc=$?\npython3 -c '+shlex.quote(_REPORT)
           +' "$harness_check_log" "$harness_check_rc"\nexit "$harness_check_rc"')
    result=environment.execute({'command':'bash -c '+shlex.quote(shell)})
    return {'output':str(result.get('output',''))[:4000], 'returncode':int(result.get('returncode',1)), 'exception_info':str(result.get('exception_info',''))}
