"""Bounded reminder after consecutive successful custom inspection turns."""
HOOK_NAME = 'inspection_checkpoint'
WORKTREE_REMAINING_TURNS = ()
PROMPT = '<factual delta> Consolidate the current hypothesis and the single unresolved observation in a few words. For related source windows, inspect_windows can bundle 2–4 reads. Follow the evidence; this reminder does not establish correctness or require an edit or submission.'
SMOKE_EVENTS = [dict(tool_name='read_file', returncode=0, step=i,
                     last_action_in_turn=True) for i in range(1, 6)]


def observe(event, state):
    step = event.get('step')
    if type(step) is not int or state.get('warnings', 0) >= 2:
        return ''
    if event.get('tool_name') not in {'read_file', 'grep', 'inspect_windows'} or event.get('returncode') != 0:
        state['steps'] = []
        state['warned_streak'] = False
        return ''
    steps = state.setdefault('steps', [])
    if step not in steps:
        steps.append(step)
    state['steps'] = steps[-5:]
    if len(state['steps']) < 5 or state.get('warned_streak') or not event.get('last_action_in_turn', True):
        return ''
    state['warnings'] = state.get('warnings', 0) + 1
    state['warned_streak'] = True
    return PROMPT.replace('<factual delta>', 'Five consecutive successful inspection tool turns have completed.')
