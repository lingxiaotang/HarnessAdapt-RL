"""Bounded feedback on repeated observable operations, including native bash."""
import hashlib
HOOK_NAME = 'progress_guard'
WORKTREE_REMAINING_TURNS = ()
PROMPT = '<factual delta> Change the next observation or correct the invocation; repeating unchanged evidence is unlikely to resolve the issue.'
SMOKE_EVENTS = [dict(tool_name='bash', normalized_call='same', output='missing module', returncode=1, step=i) for i in (1, 2, 3)]
def observe(event, state):
    step = event.get('step')
    call, output = event.get('normalized_call'), event.get('output')
    if type(step) is not int or not call or not output or state.get('warnings', 0) >= 2:
        return ''
    # Do not treat edits or artifact generation as a repeated-read problem.
    if event.get('tool_name') in ('edit_file', 'write_file', 'prepare_patch') and event.get('returncode') == 0:
        state['history'] = []
        return ''
    signature = hashlib.sha256((call+'\n'+str(event.get('returncode'))+'\n'+output).encode()).hexdigest()
    history = [(s, h) for s, h in state.get('history', []) if 0 <= step-s <= 12]
    if (step, signature) not in history: history.append((step, signature))
    state['history'] = history[-48:]
    seen_steps = {s for s, h in history if h == signature}
    if len(seen_steps) < 3 or signature in state.get('warned', []):
        return ''
    state.setdefault('warned', []).append(signature)
    state['warnings'] = state.get('warnings', 0) + 1
    return PROMPT.replace('<factual delta>', 'The same command/tool call produced identical output on at least three distinct turns within the last 12 turns. Check whether this repeats a source read, a runner/setup failure, or a failing hypothesis.')
