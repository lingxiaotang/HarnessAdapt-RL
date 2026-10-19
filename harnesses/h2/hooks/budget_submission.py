"""Budget stages use actual remaining context and turns, never a fixed step."""
HOOK_NAME = 'budget_submission'
USE_CONTEXT_BUDGET = True
WORKTREE_REMAINING_TURNS = ()
PROMPT = '<factual delta> Use observed evidence; budget pressure does not prove the fix is correct.'
SMOKE_EVENTS = [
    {'remaining_turns': 100, 'context_remaining_tokens': 25000},
    {'remaining_turns': 99, 'context_remaining_tokens': 24000},
    {'remaining_turns': 99, 'context_remaining_tokens': 24000},
]
def observe(event, state):
    if not event.get('last_action_in_turn', True):
        return ''
    turns, tokens = event.get('remaining_turns'), event.get('context_remaining_tokens')
    if type(turns) is int and turns <= 0:
        return ''
    level = 0
    for i, (turn_threshold, token_threshold) in enumerate(((30, 24576), (12, 12288), (5, 6144)), 1):
        if ((type(turns) is int and 0 < turns <= turn_threshold)
                or (type(tokens) is int and tokens <= token_threshold)):
            level = i
    if level <= state.get('level', 0):
        return ''
    state['level'] = level
    facts = []
    if type(turns) is int: facts.append(f'Remaining turns: {turns}.')
    if type(tokens) is int: facts.append(f'Remaining context before this reminder: {tokens} tokens.')
    guidance = {
        1: 'Narrow the investigation to the current hypothesis. Reuse a short reproducer; stop large reads, repeated setup attempts and full-suite runs. A focused existing test file for the changed code remains worthwhile. This is a conservation warning, not a request to submit.',
        2: 'Reserve the remaining context for one focused check and delivery. If the intended behavior is supported, use prepare_patch with the exact changed source files and review its preview; do not restart broad exploration.',
        3: 'Leave room for explicit submission. If the current repair is supported, call prepare_patch if needed, review its preview, then use a separate bash call: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt. If source changed since preparation, regenerate. Do not hand-write a patch or append fallback output.'
    }
    return PROMPT.replace('<factual delta>', ' '.join(facts) + ' ' + guidance[level])
