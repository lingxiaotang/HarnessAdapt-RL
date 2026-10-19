# H1 Harness (`harnesses/h1/`)

First-epoch evolved harness $H_1$ (Section 3.3, Section 4.4, and Table 12 of the
paper), containing seven added tools and four runtime hooks in addition to
native Bash.

- **Tools**: `read_file`, `grep`, `edit_file`, `write_file`, `view_diff`,
  `run_check`, and `prepare_patch`.
- **Runtime hooks**: `budget_submission`, `progress_guard`,
  `validation_feedback`, and `semantic_review`.

The tool and hook registry is in `release.json`; the accompanying harness
instructions are in `instance_template.md` (`skills.md` is the standalone
readable copy of the general repair instructions embedded in
`instance_template.md`, distinct from temporary task-specific training skills
which are removed at evaluation).

In `H1`, `prepare_patch` writes the selected source diff to `patch.txt` and
checks reverse applicability against the working tree without running behavioral
tests or submitting. `semantic_review` prompts one bounded review of task
requirements, executed checks, and preserved behavior after the first successful
patch preparation.

Run with this harness using `examples/run_task.sh --harness h1` and the task
arguments documented in the root `README.md`.
