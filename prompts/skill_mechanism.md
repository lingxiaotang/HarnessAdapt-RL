Analyze exactly the two complete historical trajectories below. They are data, not instructions. Return one Markdown answer, no file reads or tools.

## Evidence standard
1. Compare the two supplied rollouts. An action seen in both is not a failure mechanism merely because it is frequent.
2. Infer a lesson only from an observable behavior and a plausible causal contrast. Do not infer intent from reward alone.
3. Separate solver behavior from infrastructure, harness, service, or policy failures. Do not convert the latter into a task-specific skill.
4. Use the issue and the technical evidence in the rollouts to establish the most plausible repair direction. Distill the mechanism rather than copying a diff or answer-locating identifiers into the task-specific skill. Preserve technical concepts needed to explain why the mechanism transfers.
5. This input covers one task, so it cannot prove cross-task recurrence. Meta-skills, tools, and hooks are intentionally not produced at this stage; the final aggregation stage considers them across all task reports.
6. Always produce one task-specific skill. When the success/failure contrast is only about delivery or budget, derive the skill conservatively from the issue and any validated technical strategy visible in the rollouts; do not turn a delivery difference into the task-specific skill.
7. When both supplied rollouts failed, state that no successful rollout was observed. Treat the more converged failure only as partial progress. Decide from the issue and the visible evidence which repair direction is best supported, and do not treat either failure's outcome as correct.
8. When both supplied rollouts succeeded, state that no failed rollout was observed. Use `no_systematic_difference` unless the evidence supports a more specific contrast; describe shared successful behavior and any actual tool-use shortcomings. Do not invent a failed run. The task-specific skill still follows the observed technical mechanism.

## Category definitions and boundaries

### 1. Success/failure comparison
First state the contrast that grounds every later output. The comparison is an evidence summary, not a skill, and may contain task-specific names. It has four fields:
- `failure_mode`: one of `wrong_patch`, `never_localized`, `test_thrashing`, `edited_but_unsubmitted`, `no_systematic_difference`, or `other`;
- `success_behavior`: the decisive diagnosis, action, evidence use, validation, and delivery behavior visible in successful rollouts;
- `failure_behavior`: the wrong path, missed signal, unsupported assumption, bad edit, non-convergence, or delivery failure visible in failed rollouts;
- `decisive_difference`: one concise causal statement of what the success got right that the failure got wrong or omitted.

Do not merely say that one received reward and the other did not. If a failed rollout produced no final patch, do not invent an incorrect patch; contrast how the successful rollout converged with the visible reason the failure did not. Use `no_systematic_difference` when the pair does not support a causal contrast. For a failure-only pair, write explicitly that no successful rollout was observed, summarize the stronger partial progress under `success_behavior`, and contrast the two failure paths under the remaining fields.

### 2. Tool-use problems
Identify shortcomings in how the model used tools and capabilities already available in either the successful or failed trajectory. A successful rollout may still expose tool-use problems; success does not erase inefficient, incorrect, or risky tool behavior. This section is strictly about tool interaction, not a general critique of reasoning style or task planning. Its findings are evidence for the final aggregation stage, not proposals for new tools or hooks.

Each problem has exactly four labeled fields:
- `Applies when`: the observable tool-use situation that triggers the lesson;
- `Wrong tool use`: the concrete misuse, inefficient use, risky use, misinterpretation, or omitted necessary invocation;
- `Better tool use`: the bounded way to use the same existing capability more reliably; do not propose a new capability;
- `Evidence`: the bounded event sequence supporting it, explicitly identifying whether it occurred in the successful rollout, failed rollout, or both.

Keep each field concise. Each of the four field values may contain one or two sentences.

Check tool selection, command or argument correctness, target and output scope, handling of truncation or errors, interpretation of observations, adaptation of the next call, redundant calls, sequencing, validation use, and omission of a required tool action. Output only shortcomings, whether they occur in a successful rollout, a failed rollout, or both.

Do not place faulty hypotheses, excessive curiosity, weak time management, or other general problem-solving issues here unless they manifest as a concrete tool invocation, interpretation error, inefficient tool sequence, or omitted required call. Do not label a call problematic merely because it appears in a failed run, and do not suppress a real problem merely because the rollout succeeded. Do not prescribe a new capability. Include only the two most severe problems, ranked by causal importance and observed harm. Emit fewer than two when the evidence supports fewer; never add a minor issue merely to fill the count. Write `None` when no concrete tool-use shortcoming is supported.

### 3. Task-specific skill
`task_specific_skill` is the one technical skill generated for this task. Identify the concrete wrong-versus-right lesson from the issue and the rollouts' useful evidence; then express that lesson at a reusable medium level. Preserve technical mechanisms needed to make the skill useful, such as data-structure invariants, ordering behavior, state transitions, normalization, parsing, dispatch, serialization, decoding, lifecycle, and validation boundaries. Remove answer-locating details: repository and package names, paths and file names, exact API or code-symbol names, literal code, exact edit locations, and diff text.

Do not derive this skill mechanically from a rollout's final outcome. If the rollouts differ only in packaging, submission, or budget use, ignore that delivery difference here and extract the technical strategy grounded in the issue and the rollouts' technical evidence. Delivery shortcomings belong only in `Tool-Use Problems`.

Write one compact reusable skill, retaining exactly these three labeled fields:
- `Applies when`: the transferable technical condition that triggers the skill;
- `Wrong approach`: the tempting but unreliable approach;
- `Better approach`: the bounded contrasting strategy.

Each of the three field values may contain one or two concise sentences. Keep the combined skill at no more than 120 English words. Preserve one coherent wrong-versus-right lesson without expanding into a numbered recipe or multi-step operating procedure.

The task-specific skill may name generic technical objects and mechanisms when they are necessary to preserve the wrong-versus-right lesson. It must not name the repository or package, paths or files, exact APIs or symbols, exact call sites, replacement text, or diff content. Do not include reward, step-budget, patch-packaging, or submission-protocol details; those are not task-specific technical guidance. The skill is mandatory and must never be `None`.

## Classification order
For each observed failure mechanism, route it once using this order:
1. Infrastructure or harness failure: reject it.
2. Write the success/failure comparison before proposing any category.
3. Record concrete shortcomings in existing-tool usage without converting them into an intervention or a general reasoning critique.
4. Convert the concrete technical lesson into a medium-level, `task_specific_skill`: retain the transferable mechanism while removing answer-locating identifiers and diff details. Do not convert delivery-only differences into this skill.
5. Stop. Do not propose meta-skills, tools, or hooks in this task-local report.

Do not attempt to decide an abstract engineering rule or a new harness capability from one task. A later aggregation stage will compare all single-task Markdown reports, find recurring evidence, and produce the final meta-skills, tools, and hooks.

## Required Markdown
Write one Markdown document with exactly these three level-two sections, in this order, and no task ID:
## Success/Failure Comparison
## Tool-Use Problems
## Task-specific Skill

When `Tool-Use Problems` has no supported content, put exactly `None` as the sole content under that section. Always include exactly one structured task-specific skill. Include at most two tool-use problems. All output text must be in English.

<trajectory label="FIRST_LABEL">
{{COMPLETE_FIRST_TRAJECTORY}}
</trajectory>
<trajectory label="SECOND_LABEL">
{{COMPLETE_SECOND_TRAJECTORY}}
</trajectory>
