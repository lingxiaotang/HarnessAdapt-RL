"""Frozen 1,058-task two-epoch schedule and 512-task Epoch 2 assisted subset (Appendix A.1 & B.1)."""
from __future__ import annotations

import copy
import random
from pathlib import Path

from .config import read_json


def make_annealing_plan(
    labels,
    batch_size: int = 46,
    seed: int = 20260720,
    half_assisted_ids=(),
    *,
    epochs: int = 2,
    assisted_subset_size: int = 512,
):
    """Build the two-epoch 46-task-per-update schedule over the 1,058-task pool.

    Epoch 1 (updates 1-23) assists all 1,058 tasks.
    Epoch 2 (updates 24-46) assists the fixed 512-task subset.
    """
    labels = list(labels)
    half = set(half_assisted_ids)
    if len(set(labels)) != len(labels) or len(labels) != 1058 or batch_size != 46:
        raise ValueError("Schedule requires the 1,058 unique tasks with batch_size=46 (23 updates/epoch)")
    if len(half) != assisted_subset_size or not half <= set(labels):
        raise ValueError(f"Epoch 2 assisted subset must contain exactly {assisted_subset_size} tasks from the pool")

    batches, assisted, stages = [], [], []
    steps_per_epoch = len(labels) // batch_size
    for epoch, stage in enumerate(("full_assistance", "half_assistance")[:epochs]):
        order = list(labels)
        random.Random(seed + epoch).shuffle(order)
        for offset in range(0, len(order), batch_size):
            ids = order[offset : offset + batch_size]
            batches.append(ids)
            assisted.append(ids if epoch == 0 else [x for x in ids if x in half])
            stages.append(stage)
    return {
        "sampling": "epoch_shuffled_full_half",
        "seed": seed,
        "pool_tasks": len(labels),
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": epochs * steps_per_epoch,
        "batches": batches,
        "stages": stages,
        "assisted_task_ids": assisted,
        "half_assisted_task_ids": [x for x in labels if x in half],
        "stage_unique_tasks": [len(labels)] * epochs,
        "stage_assisted_tasks": [len(labels), len(half)][:epochs],
    }


def make_native_plan(labels, batch_size: int = 46, epochs: int = 2, seed: int = 20260720):
    """Two-epoch schedule without skill assistance (used by GRPO, OpenHands-GRPO, Frozen-H1)."""
    labels = list(labels)
    if len(set(labels)) != len(labels) or len(labels) % batch_size or epochs <= 0:
        raise ValueError("Native epochs require a unique pool and complete equal-sized batches")
    batches = []
    for epoch in range(epochs):
        order = list(labels)
        random.Random(seed + epoch).shuffle(order)
        batches.extend(order[i : i + batch_size] for i in range(0, len(order), batch_size))
    return {
        "sampling": "epoch_shuffled_native",
        "seed": seed,
        "pool_tasks": len(labels),
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": len(labels) // batch_size,
        "total_steps": len(batches),
        "batches": batches,
        "stages": ["no_assistance"] * len(batches),
        "assisted_task_ids": [[] for _ in batches],
        "half_assisted_task_ids": [],
        "stage_unique_tasks": [len(labels)] * epochs,
        "stage_assisted_tasks": [0] * epochs,
    }


def plan_for_config(labels, config: dict, *, half_assisted_ids=()):
    if config.get("sampling_schedule") == "epoch_shuffled_native":
        return make_native_plan(
            labels,
            config["tasks"],
            config["epochs"],
            config["selection_seed"],
        )
    return make_annealing_plan(
        labels,
        config["tasks"],
        config["selection_seed"],
        half_assisted_ids,
        epochs=config["epochs"],
        assisted_subset_size=config.get("epoch2_assisted_tasks", 512),
    )


def assisted_ids(config: dict, rollout_id: int) -> list[str]:
    if config.get("sampling_schedule") == "epoch_shuffled_native":
        return []
    return read_json(Path(config["sampling_plan"]))["assisted_task_ids"][rollout_id]


def planned_ids(config: dict, rollout_id: int) -> list[str]:
    return read_json(Path(config["sampling_plan"]))["batches"][rollout_id]


def optimizer_batch_size(root: Path, rollout_id: int, nominal: int) -> int:
    config = read_json(root / "config.json")
    return len(planned_ids(config, rollout_id)) * config["rollouts_per_task"]


class PlannedBatch:
    """Consume only the frozen update's tasks while isolating preparation/comparison cursors."""

    def __init__(self, source, ids, rollout_id: int, config: dict, *, training: bool):
        self.source = source
        self.ids = ids
        self.used = 0
        self.n = config["rollouts_per_task"]
        self.pool_size = config["pool_tasks"]
        self.offset = rollout_id * config.get("tasks", len(ids))
        self.rows = {str(s.label): s for s in source.dataset.samples}
        if len(self.rows) != self.pool_size or not set(ids) <= self.rows.keys():
            raise ValueError("Runtime dataset differs from the frozen 1,058-task pool")
        if training and (
            source.sample_group_index != self.offset
            or source.sample_index != self.offset * self.n
        ):
            raise ValueError("Checkpoint sampling counters do not match the requested training step")
        if not training:
            source.sample_group_index = self.offset
            source.sample_index = self.offset * self.n

    def get_samples(self, count: int):
        if count <= 0 or self.used + count > len(self.ids):
            raise ValueError("Sampling exceeded the frozen batch")
        groups = []
        for label in self.ids[self.used : self.used + count]:
            group = []
            for _ in range(self.n):
                sample = copy.deepcopy(self.rows[label])
                sample.group_index = self.source.sample_group_index
                sample.index = self.source.sample_index
                self.source.sample_index += 1
                group.append(sample)
            self.source.sample_group_index += 1
            groups.append(group)
        self.used += count
        absolute = self.offset + self.used
        self.source.epoch_id, self.source.sample_offset = divmod(absolute, self.pool_size)
        return groups

    def add_samples(self, samples):
        if samples:
            raise ValueError("Unexpected buffered samples in a frozen batch")
