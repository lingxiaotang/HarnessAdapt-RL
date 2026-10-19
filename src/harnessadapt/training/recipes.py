"""Training recipes matching all experimental configurations in Section 4.1, Table 1, and Appendix B/C."""
from __future__ import annotations

from .config import DEFAULTS, PACKAGE_ROOT
from harnessadapt.openhands import BACKEND as OPENHANDS_BACKEND, VERSION as OPENHANDS_VERSION

# Main paper configurations (Section 4.1 & Table 1)
GRPO_RECIPE = "grpo"
OPENHANDS_GRPO_RECIPE = "openhands_grpo"
SKILL_HINT_RECIPE = "skill_hint"
SKILL_MECHANISM_RECIPE = "skill_mechanism"
SKILL_BANK_RECIPE = "skill_bank"
SKILL_GATED_RECIPE = "skill_gated"
PRE_EVOLVED_RECIPE = "pre_evolved"
H1_MECHANISM_RECIPE = "h1_mechanism"
HARNESSADAPT_RL_RECIPE = "harnessadapt_rl"

# Additional frozen-harness reuse configurations (Section 4.6 & Appendix C.6)
FROZEN_H1_RECIPE = "frozen_h1"
FROZEN_H1_MECHANISM_RECIPE = "frozen_h1_mechanism"
FROZEN_H2_MECHANISM_RECIPE = "frozen_h2_mechanism"

ALL_RECIPES = (
    GRPO_RECIPE,
    OPENHANDS_GRPO_RECIPE,
    SKILL_HINT_RECIPE,
    SKILL_MECHANISM_RECIPE,
    SKILL_BANK_RECIPE,
    SKILL_GATED_RECIPE,
    PRE_EVOLVED_RECIPE,
    H1_MECHANISM_RECIPE,
    HARNESSADAPT_RL_RECIPE,
    FROZEN_H1_RECIPE,
    FROZEN_H1_MECHANISM_RECIPE,
    FROZEN_H2_MECHANISM_RECIPE,
)


def validate_recipe(config: dict) -> None:
    """Reject obsolete compatible bundles and silent fallback to Mini-SWE."""
    if config.get("recipe") == OPENHANDS_GRPO_RECIPE or config.get("harness_profile") == "openhands":
        if config.get("backend") != OPENHANDS_BACKEND or config.get("openhands_version") != OPENHANDS_VERSION:
            raise ValueError("OpenHands requires the pinned official SDK backend; prepare a fresh run")
        if config.get("fixed_harness_source") or config.get("verified_evolved_harness"):
            raise ValueError("Official OpenHands cannot use a local evolved/compatible harness bundle")
        if any(config.get(key) for key in ("skill_variant", "evolve_epoch1", "evolve_epoch2", "assisted_steps", "pre_evolve_on_base")):
            raise ValueError("OpenHands baseline must not enable teacher skills or harness evolution")


def recipe_defaults(recipe: str | None = None) -> dict:
    values = dict(DEFAULTS)
    if recipe is None:
        recipe = HARNESSADAPT_RL_RECIPE
    if recipe not in ALL_RECIPES:
        raise ValueError(f"Unknown training recipe: {recipe!r}. Valid recipes: {ALL_RECIPES}")

    # Shared two-epoch, 46-update schedule across all configurations (Table 5 & Appendix B.1)
    values.update(
        recipe=recipe,
        tasks=46,
        pool_tasks=1058,
        epochs=2,
        steps_per_epoch=23,
        total_steps=46,
        full_assisted_steps=23,
        assisted_steps=46,
        epoch2_assisted_tasks=512,
        rollouts_per_task=4,
        probe_rollouts_per_task=4,
        turns=60,
        context_tokens=65536,
        temperature=1.0,
        learning_rate=1e-6,
        save_interval=5,
        periodic_verified=True,
        annealing_schedule="epoch_full_half",
        skill_generation_schedule="regenerate_each_epoch",
        fixed_harness_source=None,
        evolve_epoch1=False,
        evolve_epoch2=False,
        evolution_rounds_per_epoch=5,
        skill_variant=None,
    )

    if recipe == GRPO_RECIPE:
        # GRPO baseline: H0, no skills, 8 rollouts per task (368 per update, 16,928 total)
        values.update(
            rollouts_per_task=8,
            probe_rollouts_per_task=0,
            assisted_steps=0,
            full_assisted_steps=0,
            epoch2_assisted_tasks=0,
            annealing_schedule=None,
            sampling_schedule="epoch_shuffled_native",
            static_miniswe=True,
            skill_teacher_backend=None,
            skill_teacher_model=None,
            harness_teacher_backend=None,
            harness_teacher_model=None,
            add_turn_reminder=True,
            native_add_turn_reminder=True,
        )
    elif recipe == OPENHANDS_GRPO_RECIPE:
        # Official upstream Agent/Conversation and tools, isolated from Slime's
        # Python environment.
        values.update(
            probe_rollouts_per_task=0,
            assisted_steps=0,
            full_assisted_steps=0,
            epoch2_assisted_tasks=0,
            annealing_schedule=None,
            sampling_schedule="epoch_shuffled_native",
            harness_profile="openhands",
            backend=OPENHANDS_BACKEND,
            openhands_version=OPENHANDS_VERSION,
            verified_evolved_harness=False,
            skill_teacher_backend=None,
            skill_teacher_model=None,
            harness_teacher_backend=None,
            harness_teacher_model=None,
        )
    elif recipe == SKILL_HINT_RECIPE:
        # Skill-Hint: H0 + near-answer hints
        values.update(
            skill_variant="hint",
            static_miniswe=True,
            harness_teacher_backend=None,
            harness_teacher_model=None,
        )
    elif recipe == SKILL_MECHANISM_RECIPE:
        # Skill-Mechanism: H0 + mechanism-level skills
        values.update(
            skill_variant="mechanism",
            static_miniswe=True,
            harness_teacher_backend=None,
            harness_teacher_model=None,
        )
    elif recipe == SKILL_BANK_RECIPE:
        # Skill-Bank: H0 + cross-task skill bank (at most 10 entries, <=1000 words)
        values.update(
            skill_variant="bank",
            static_miniswe=True,
            harness_teacher_backend=None,
            harness_teacher_model=None,
        )
    elif recipe == SKILL_GATED_RECIPE:
        # Skill-Gated (Appendix C.5): H0 + mechanism-level skills only when all 4 preparation attempts fail
        values.update(
            skill_variant="mechanism_gated",
            static_miniswe=True,
            harness_teacher_backend=None,
            harness_teacher_model=None,
        )
    elif recipe == PRE_EVOLVED_RECIPE:
        # Pre-Evolved: H_pre discovered on base model before policy training, then fixed with mechanism skills
        values.update(
            skill_variant="mechanism",
            pre_evolve_on_base=True,
            keep_harness_all_epochs=True,
            verified_evolved_harness=True,
        )
    elif recipe == FROZEN_H1_RECIPE:
        # Frozen-H1 (Appendix C.6): H1 fixed from update 1, no skills
        values.update(
            probe_rollouts_per_task=0,
            assisted_steps=0,
            full_assisted_steps=0,
            epoch2_assisted_tasks=0,
            annealing_schedule=None,
            sampling_schedule="epoch_shuffled_native",
            fixed_harness_source=str(PACKAGE_ROOT / "harnesses" / "h1"),
            keep_harness_all_epochs=True,
            verified_evolved_harness=True,
            skill_teacher_backend=None,
            skill_teacher_model=None,
            harness_teacher_backend=None,
            harness_teacher_model=None,
        )
    elif recipe == FROZEN_H1_MECHANISM_RECIPE:
        # Frozen-H1+Mechanism (Appendix C.6): H1 fixed from update 1 + mechanism-level skills
        values.update(
            skill_variant="mechanism",
            fixed_harness_source=str(PACKAGE_ROOT / "harnesses" / "h1"),
            keep_harness_all_epochs=True,
            verified_evolved_harness=True,
            harness_teacher_backend=None,
            harness_teacher_model=None,
        )
    elif recipe == H1_MECHANISM_RECIPE:
        # H1+Mechanism (Section 4.1): evolves H0 -> H1 in Epoch 1, keeps H1 fixed in Epoch 2 + mechanism skills
        values.update(
            skill_variant="mechanism",
            evolve_epoch1=True,
            evolve_epoch2=False,
            keep_harness_all_epochs=True,
            verified_evolved_harness=True,
        )
    elif recipe == FROZEN_H2_MECHANISM_RECIPE:
        # Frozen-H2+Mechanism (Section 4.6): H2 fixed from update 1 + mechanism-level skills
        values.update(
            skill_variant="mechanism",
            fixed_harness_source=str(PACKAGE_ROOT / "harnesses" / "h2"),
            keep_harness_all_epochs=True,
            verified_evolved_harness=True,
            harness_teacher_backend=None,
            harness_teacher_model=None,
        )
    elif recipe == HARNESSADAPT_RL_RECIPE:
        # HarnessAdapt-RL (Full method): evolves H0 -> H1 in Epoch 1 and H1 -> H2 in Epoch 2 + mechanism skills
        values.update(
            skill_variant="mechanism",
            evolve_epoch1=True,
            evolve_epoch2=True,
            keep_harness_all_epochs=True,
            verified_evolved_harness=True,
        )
    return values
