"""Slime Ray GRPO training driver with audited parameter updates and checkpoint saving."""
from __future__ import annotations

import os
from pathlib import Path

import ray
from slime.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()
    release_train = args.release_train
    eval_steps = getattr(args, "eval_steps", None)
    eval_enabled = args.eval_interval is not None or eval_steps is not None

    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_cls = None
    if os.environ.get("R2E_COEVOLVE_ROOT") and not args.debug_rollout_only:
        from harnessadapt.training.actor import AuditedActor

        actor_cls = AuditedActor
    actor_model, critic_model = create_training_models(
        args, pgs, rollout_manager, actor_cls=actor_cls
    )

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if args.num_rollout == 0 and eval_enabled:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    stop_after = int(os.environ.get("R2E_STOP_AFTER_STEP", args.num_rollout))
    if args.num_rollout > 0 and not args.start_rollout_id <= stop_after <= args.num_rollout:
        raise ValueError("Training barrier is outside the remaining optimizer schedule")

    for rollout_id in range(args.start_rollout_id, stop_after):
        schedule_rollout_id = rollout_id - args.start_rollout_id
        schedule_num_rollout = args.num_rollout - args.start_rollout_id
        if "R2E_STOP_AFTER_STEP" in os.environ:
            schedule_rollout_id = rollout_id
            schedule_num_rollout = stop_after
        if eval_enabled and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(
                    actor_model.async_train(
                        rollout_id, rollout_data_ref, external_data=value_refs
                    )
                )
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        checkpoint_saved = release_train or should_run_periodic_action(
            schedule_rollout_id,
            args.save_interval,
            num_rollout_per_epoch,
            schedule_num_rollout,
        )
        if checkpoint_saved:
            force_sync = release_train or rollout_id == stop_after - 1
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if os.environ.get("R2E_COEVOLVE_ROOT") and not args.debug_rollout_only:
            from harnessadapt.training.config import read_json, write_json

            root = Path(os.environ["R2E_COEVOLVE_ROOT"])
            probes = [
                read_json(p)
                for p in (root / "steps" / f"{rollout_id + 1:04d}").glob(
                    "parameter_update_rank_*.json"
                )
            ]
            if not probes or not any(p["changed_values"] > 0 for p in probes):
                raise RuntimeError("Cannot verify nonzero parameter update")
            write_json(
                root / "steps" / f"{rollout_id + 1:04d}" / "optimizer_complete.json",
                {
                    "step": rollout_id + 1,
                    "checkpoint": args.save if checkpoint_saved else None,
                    "checkpoint_saved": checkpoint_saved,
                    "optimizer_call_completed": True,
                    "inference_weights_synchronized": True,
                    "nonzero_parameter_update_observed": True,
                },
            )
            write_json(
                root / "steps" / f"{rollout_id + 1:04d}" / "progress.json",
                {"step": rollout_id + 1, "stage": "optimizer_complete"},
            )
        if should_run_periodic_action(
            schedule_rollout_id,
            args.eval_interval,
            num_rollout_per_epoch,
            schedule_num_rollout,
            explicit_steps=eval_steps,
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
