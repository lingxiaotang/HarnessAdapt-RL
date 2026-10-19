"""One advisory semantic checkpoint, with no grader/teacher or submission gate."""
HOOK_NAME='semantic_review'
USE_CONTEXT_BUDGET=True
WORKTREE_REMAINING_TURNS=()
PROMPT='<factual delta> This is a bounded review, not a demand for exhaustive proof or a submission gate.'
SMOKE_EVENTS=[{'tool_name':'prepare_patch','returncode':2,'context_remaining_tokens':20000},
              {'tool_name':'prepare_patch','returncode':0,'context_remaining_tokens':19000},
              {'tool_name':'prepare_patch','returncode':0,'context_remaining_tokens':18000}]
def observe(event,state):
    if event.get('tool_name')!='prepare_patch' or event.get('returncode')!=0 or state.get('reviewed'):
        return ''
    state['reviewed']=True
    tokens=event.get('context_remaining_tokens');turns=event.get('remaining_turns')
    low=(type(tokens)is int and tokens<=6144) or (type(turns)is int and turns<=5)
    text=('Check the evidence behind this prepared patch once, in a few short statements: '
          '(1) the observable behavior requested by the ORIGINAL issue, including explicitly named variants; '
          '(2) the actual edited repository entry point invoked by your check and the requirement-derived assertion that executed; '
          '(3) one relevant behavior preserved, plus any unresolved contradictory observation. '
          'A reimplementation of the fix in a scratch script, uncalled test functions, or assertions derived from current output do not validate the patch. '
          'For state changes, consider the next operation; for external interfaces, distinguish confirmed support from guessed arguments. '
          'Use this to identify at most one concrete gap, not to recite confidence or invent new requirements. ')
    text+=('Budget is tight: use existing evidence; do not start broad checks. Retain uncertainty and finish with the explicit bash submission when ready.' if low else
           'If a concrete gap remains, make one discriminating observation on the real code path and address it. Otherwise finish with the explicit bash submission. Do not repeat this review.')
    return PROMPT.replace('<factual delta>',text)
