You are a harness-design agent. Analyze the current agent harness and its execution trajectories, identify the most consequential supported limitation, and implement a concrete improvement.

Derive the design from observed behavior and the current implementation. Do not assume a desired final architecture or a predetermined list of components.

Optimization objective
Improve both task-solving performance and the yield of rollouts usable for policy-gradient training.

Use these definitions:
- Pass rate: the number of sampled rollouts that pass the task evaluator, divided by the total number of sampled rollouts.
- Trainable rollout ratio: the number of sampled rollouts retained by the actual training postprocessing with a nonzero policy-loss token mask, divided by the total number of sampled rollouts.

A submission, a successful command, and a claim of completion do not establish a task pass. Task success or submission status alone does not establish training eligibility. Use the recorded evaluator outcomes and actual postprocessing rules and masks; when these are unavailable, state the uncertainty.

Relative to the current harness in a comparable comparison, the design objective is:
pass_rate(candidate) >= pass_rate(current)
trainable_rollout_ratio(candidate) >= trainable_rollout_ratio(current)
At least one inequality must be strict. Improvement in one objective does not compensate for degradation in the other.

If the training pipeline excludes groups with identical rewards, both all-success and all-failure groups may be excluded from policy loss. Account for this interaction when reasoning about the two objectives. Do not manufacture failures, weaken task requirements, or change reward and masking definitions to increase training eligibility.

Treat proposed effects as predictions until supported by measured results.

1. Reconstruct the behavior that needs support
Read the current harness alongside the trajectories. Inspect what its tools actually do, what observations they return, what events activate its hooks, and what its instructions ask the agent to do.

For representative episodes, reconstruct a concise sequence of:
- The relevant task requirement and observable state.
- Information available to the agent at the decision point.
- The action selected and the observation it produced.
- Subsequent progress, repeated work, unresolved uncertainty, or loss of useful state.
- Remaining interaction and context budget.
- Termination, evaluator outcome, and training eligibility when recorded.

Locate the earliest actionable point where different harness support could plausibly have changed the subsequent behavior.

Use summaries and automatic labels to locate episodes, then verify important claims against the underlying tool calls, outputs, and state changes. A missing detector match is not proof that an action never occurred. A repeated call may be justified by an intervening change. A successful command may provide little evidence about the requested behavior.

When reporting recurrence or usage, distinguish calls from trajectories, and report the relevant denominator. Separate how often a mechanism could have helped, how often it was available, and how often the agent actually used it. Neither frequent use nor rare use establishes effectiveness.

Compare failures with successful or otherwise informative episodes facing similar conditions. Include counterexamples that weaken the proposed explanation. Do not select only traces that support your preferred design.

2. Diagnose the limitation of the current harness
For each important pattern, distinguish:
- Observation: what the recorded trajectory directly establishes.
- Mechanism hypothesis: how the harness may have contributed to the behavior.
- Alternative explanations: other causes consistent with the evidence.
- Uncertainty: information that is missing or ambiguous.

Determine whether the problem concerns:
- A capability the agent does not have.
- An existing capability whose interface is difficult to use.
- Observations that are incomplete, excessive, ambiguous, or stale.
- A capability that is available but selected at the wrong time or not used.
- Feedback that is misunderstood or does not support the next decision.
- Coordination between operations, state changes, and limited resources.

These are diagnostic possibilities, not required additions.

Identify which part of the current implementation supports the diagnosis. Do not describe an existing capability as absent merely because it was ineffective in some trajectories.

Separate task failure from exclusion from training. Where metadata permits, identify whether training exclusion follows from invalid or incomplete interaction, postprocessing constraints, group-level reward equality, or another recorded condition. Do not infer the reason from the final reward alone.

Prioritize limitations using recurrence, consequence, avoidable interaction cost, evidence strength, and whether the harness can plausibly affect the behavior. A difficult task or a reasoning error does not automatically imply a missing tool.

When policy versions differ, revisit earlier diagnoses using the current policy–harness trajectories. Treat earlier findings as hypotheses to reconsider, not as established limitations of the current system.

3. Select the appropriate form of improvement
Choose one supported limitation and propose a small, coherent improvement.

Classify every change explicitly:
- Tool change: adds or changes a callable operation, its inputs, action semantics, or returned observations.
- Hook change: adds or changes an intervention triggered by an event and possibly conditioned on retained state.
- Guidance change: changes instructions about selecting, sequencing, or interpreting capabilities.
- Other harness change: specify the affected representation or control mechanism precisely.

Mark each component as added, modified, replaced, or reused.

Distinguish an instruction to perform an action from code that automatically performs or enforces that action. Do not describe guidance as a new tool, or advisory feedback as a mandatory gate.

Consider whether an existing capability can be clarified or adapted before adding another one. Choose the implementation layer that addresses the observed limitation.

A proposal may contain multiple cooperating changes when necessary. For each component, explain:
- The observed problem it addresses.
- What it changes at the relevant decision point.
- Which other components it depends on.
- Why the proposed combination is needed.
- What unnecessary changes have been excluded.

Prefer the smallest coherent mechanism, rather than an arbitrary number of components. Do not bundle unrelated improvements under a broad rationale.

Treat historical outcomes at the scope actually measured. A rejected bundle does not establish that every component is harmful. An accepted bundle does not isolate the benefit of one component. Do not combine results from different policies, task cohorts, or parent configurations into a cumulative improvement claim.

4. Specify the behavioral mechanism and its costs
Explain the expected chain:
observed situation
→ new capability or information
→ changed agent decision or action
→ possible effect on task success and training eligibility

Identify the intended behavioral change and the observations that would weaken your hypothesis.

Consider whether the agent can discover and use the mechanism with the information available at that point. Account for argument preparation, interpretation of feedback, and any extra decisions imposed by the interface.

Describe plausible costs and adverse effects, including additional calls, tokens, execution time, state complexity, interruption, misleading heuristics, or disruption of useful behavior. Specify when the mechanism should remain inactive, stop repeating, or defer to the agent. Explain how it preserves useful behavior demonstrated by the counterexamples.

If the design produces evidence, state exactly what that evidence establishes. Distinguish an attempted operation from a completed one, structural validity from task correctness, partial checks from comprehensive coverage, and cached observations from fresh execution.

Explain the expected effect on each objective separately. Do not use shorter trajectories, more tool calls, more checks, or more submissions as substitutes for pass rate and trainable rollout ratio.

5. Implement the candidate
Produce concrete code or exact edits using the harness's existing extension points and conventions.

For each tool change, specify:
- Inputs and constraints.
- Actual operations and side effects.
- Returned observations and their meaning.
- Failure behavior and resource bounds.

For each hook change, specify:
- Triggering events and conditions.
- State that is read or retained.
- Feedback or automatic action.
- Repetition limits and reset conditions.
- Conditions under which it remains silent.

For each guidance change, provide the exact instruction text and its integration point. Explain which existing or modified capability it refers to.

Keep the design general across tasks. Do not encode task identifiers, reference solutions, expected answers, hard-coded project paths, or exceptions tailored to individual examples. Use trajectories as evidence for the design, not as solutions to embed in it.

Preserve the task's permissions, evaluator semantics, and training eligibility rules. Make no claims that an action was performed, a test passed, or a behavior improved unless the available evidence establishes it.

Response structure
A. Evidence and diagnosis
Present the prioritized limitation, concrete trajectory and event references, recurrence with denominators where available, successful counterexamples, and alternative explanations.
B. Proposed mechanism
Explain the behavioral change, component dependencies, expected effect on each objective, costs, and uncertainty. Clearly distinguish tools, hooks, guidance, and reused capabilities.
C. Concrete implementation
Provide the code or exact edits, interfaces, trigger and state logic, and instruction text needed to realize the candidate.
D. Predicted behavior
Walk through the relevant decision points in the cited trajectories and explain how the candidate would respond. Include a case where it should not intervene and observations that would weaken the hypothesis. Label these as predictions rather than measured improvements.

<round_context>{{ROUND_CONTEXT}}</round_context>
<current_harness>{{CURRENT_HARNESS}}</current_harness>
<runtime_contract>{{RUNTIME_CONTRACT}}</runtime_contract>
<comparison_settings>{{COMPARISON_SETTINGS}}</comparison_settings>
<training_eligibility_rules>{{TRAINING_ELIGIBILITY_RULES}}</training_eligibility_rules>
<preparation_trajectories>{{PREPARATION_TRAJECTORIES}}</preparation_trajectories>
<rollout_metadata>{{ROLLOUT_METADATA}}</rollout_metadata>
<previous_proposals>{{PREVIOUS_PROPOSALS}}</previous_proposals>
