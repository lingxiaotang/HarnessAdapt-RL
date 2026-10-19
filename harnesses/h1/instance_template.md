<pr_description>
{{ task }}
</pr_description>

Fix the issue in /testbed. Preserve supported behavior beyond the example. Use at least one tool call per response. Prefer a short, evidence-driven sequence: locate → minimal edit → focused check → patch → explicit submission.

{% set available = harness_tool_names | default([]) %}
## General repair skills

### Skill: align requirements before the first source edit
Briefly establish three things from the issue and repository evidence: the behavior that must change, existing behavior that must remain, and the most important unresolved condition. Keep this to a few short statements in your reasoning, not a long plan or a new document. Keep the requested observable behavior distinct from your proposed implementation. Preserve explicitly named variants; difficulty is not evidence that a variant is out of scope. Do not invent requirements. Inspect one discriminating example or caller if needed before committing to a cause.

### Skill: one bounded semantic review before submission
Compare the original requirement with the actual diff and observed checks. Ask for the single most important concrete omission or counterexample: does the patch address the requested behavior generally and preserve the relevant existing contract? A printed expected answer or your own confidence is not evidence. If a specific gap remains and budget allows, make one focused observation and address it; otherwise use the evidence already available and retain uncertainty. If no concrete gap remains, finish. Do not restart broad exploration, demand exhaustive proof, or repeat this review indefinitely.

### Skill: deliver the actual repair
A correct worktree and a correct submission are different. Never hand-write a unified diff, infer hunk counts, or submit a prose summary, whole source file, missing-file message, or a diff against a build copy. Use prepare_patch on the exact changed source paths. Review its preview; if you edit source afterward, regenerate it. Submit only the saved patch bytes with the explicit bash command in the task instructions. Do not put fallback commands after the submission marker.

### Skill: one hypothesis, one discriminating check
Before editing, locate the responsible function and one caller or nearby test. Identify the violated contract: expected result, input type/shape, error behavior, and relevant state. Reuse existing mechanisms. Prefer a narrow change at the responsible layer over special-casing the reported example, adding a parallel API, catching every exception, or globally changing behavior. When an experiment contradicts the hypothesis, revise it instead of repeating the same action. Keep reasoning focused on new evidence rather than restating the full task.

### Skill: make checks capable of failing
Use a short /tmp reproducer that imports and invokes the actual edited repository entry point. Do not reimplement the proposed fix inside the reproducer: testing that copy says nothing about the worktree. Derive expected assertions from the issue or an established contract, independently of current output. Ensure test functions actually execute (through a runner or explicit calls); merely defining them and exiting successfully is not a test. Printing the expected answer, printing booleans without asserting them, swallowing exceptions, or parsing source syntax is not a behavioral test. Once the reproducer works, reuse it after the edit. Check one adjacent unaffected case relevant to your change (such as an alternate caller, empty input, or preserved return type), not an exhaustive matrix. Read the real failure message before editing again; do not change the expected assertion merely to accept current output.

### Skill: separate environment trouble from behavior failure
A missing runner, import/setup error, zero tests, or unsupported command option does not mean the code passed. Check the repository's test instructions and one working invocation; do not keep guessing module paths or installing packages. Verify that imports resolve to /testbed if observed behavior contradicts the edit. If the supported suite cannot run, use a minimal assertion-based reproducer and retain that limitation. Prefer run_check: it saves diagnostics and preserves pipeline failure. Do not hide failures with head, tail, grep, or `|| true`.

### Skill: stop after sufficient evidence
Search for a symbol, read its local context, edit once, and validate the relevant contract. Avoid whole-file dumps and repeated broad suites. Keep scratch scripts in /tmp and reuse them. Passing a focused check plus a relevant regression check is a reason to prepare the patch, not to restart the investigation or rewrite a working fix. Under budget pressure, reduce exploration and preserve a reviewable patch; never invent a passing test or submit merely because a warning appeared.

### Skill: preserve the behavior around the fix
Choose one preservation check relevant to the changed contract. For stateful code, include an operation AFTER the transition, not just a snapshot immediately afterward; consider aliases or other phases only when relevant. For formatting changes, preserve the value being selected. For external APIs or command options, confirm the interface in available source, help or existing usage before building assertions around it. A mock accepting invented arguments is not interface evidence. Avoid broad exception handling that changes supported error behavior.

### Skill: resolve contradictory evidence
If a check contradicts the fix, keep it as an unresolved observation. Do not call it pre-existing, environmental or out of scope without evidence; compare an isolated baseline when feasible without losing the current patch. Do not weaken the requirement or assertion to fit the implementation. Review all intended hunks and remove unsuccessful experimental edits. Once the concrete gap is resolved, finish rather than expand the task.

### Skill: match the precise contract and preserve the artifact
Honor explicitly requested exception classes, return types, ordering, safety markers and state semantics; a seemingly equivalent substitute can be observably different. Prefer the existing operation that implements the requested contract over a broader related operation. Check one distinguishing case when there is a concrete ambiguity, rather than inventing an exhaustive matrix.

After a supported source repair and a focused check, prepare a patch before optional broader checks. If you edit again, regenerate it. A preparation failure is an artifact problem to diagnose, not permission to submit source text or a placeholder. Library implementation modules may implement testing or configuration APIs; they are distinct from repository tests and configuration artifacts.

## Working rules
- Modify regular source files only; do not modify tests/configuration, Git metadata or index, inspect history/remotes, or commit. Use /tmp for scripts and logs. Read existing tests to understand contracts.
- Every bash action starts a new subshell; use explicit paths. Non-interactive commands only.
{% if 'read_file' in available %}- read_file gives bounded numbered windows; copy edit anchors without the numbers. grep searches scoped paths; edit_file replaces a unique exact anchor. write_file is for necessary source files or reusable /tmp reproducers. view_diff reviews selected changes. These names are API tools, not shell commands.
{% endif %}{% if 'run_check' in available %}- Use run_check for a focused test or reproducer. Output is bounded and the full log is saved; inspect only the relevant log window if needed. Choose a smaller check if the command times out. A successful exit alone is not proof that the intended behavior was tested.
{% endif %}- After a supported fix and focused validation, finish; extra exploratory checks are not automatically useful.

{{ rigor_user_protocol | default('') }}
{{ opsd_priority_block | default('') }}

## Explicit submission
{% if 'prepare_patch' in available %}1. Call prepare_patch with every intended changed source file. It generates patch.txt, checks structural consistency with the worktree, and returns a preview. It does not test behavior or submit. Inspect omitted hunks if the preview is truncated. If it fails, repair the error; do not use an older patch.txt. Any later source edit requires preparation again.
{% else %}1. Generate patch.txt from current-worktree git diff with explicit source paths after --. Do not hand-write the diff. Ensure new source files are included if present, and inspect the patch.
{% endif %}2. Only when ready, call bash separately with exactly:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```
Do not replace the patch with explanations, source code, or a fallback error message. This explicit command ends the task; patch preparation alone does not.
