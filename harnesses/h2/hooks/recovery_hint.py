"""Points to restore_file when a Git revert attempt is refused by the task's command policy."""
import re
HOOK_NAME = 'recovery_hint'
USE_CONTEXT_BUDGET = False
WORKTREE_REMAINING_TURNS = ()
PROMPT = '<factual delta> Restoring discards only your edits to that file; re-apply the intended change afterwards.'
SMOKE_EVENTS = [
    {'tool_name': 'bash', 'arguments': {'command': 'git checkout -- pkg/mod.py'}, 'returncode': 126,
     'output': "SKYRL_GIT_POLICY_BLOCKED: Git subcommand 'checkout' is forbidden."},
    {'tool_name': 'bash', 'arguments': {'command': 'ls pkg'}, 'returncode': 0, 'output': 'mod.py'},
]
REVERT = re.compile(r'\bgit\s+(checkout|restore|stash|show|reset)\b', re.I)
BLOCKED = 'SKYRL_GIT_POLICY_BLOCKED'
TARGET = re.compile(r'(?:\s|:)(?:/testbed/)?((?!/|-)[\w.-]+(?:/[\w.-]+)*\.[A-Za-z]\w*)\b')


def observe(event, state):
    if event.get('tool_name') != 'bash' or state.get('fired', 0) >= 2:
        return ''
    command = str((event.get('arguments') or {}).get('command') or '')
    match = REVERT.search(command)
    if not match or BLOCKED not in str(event.get('output') or ''):
        return ''
    state['fired'] = state.get('fired', 0) + 1
    subcommand = match.group(1).lower()
    segment = re.split(r'&&|\|\||[;|>\n]', command[match.start():])[0]
    paths = [p for p in TARGET.findall(segment) if not p.startswith('tmp/')]
    target = paths[0] if paths else '<file>'
    call = 'restore_file with {"path": "%s"}' % target
    facts = 'git %s is unavailable in this task.' % subcommand
    if subcommand == 'show':
        advice = ('To see what you changed, use view_diff or git diff -- %s; your edits stay. '
                  'Only if you want to discard them, call %s.' % (target, call))
    elif subcommand == 'stash':
        advice = ('To compare with the starting behavior, copy your edited file to /tmp, call %s, '
                  'run the check, then copy your version back.' % call)
    else:
        advice = 'To return %s to its starting version, call %s. Do not rebuild a file from memory.' % (target, call)
    return PROMPT.replace('<factual delta>', facts + ' ' + advice)
