"""Observe real optimizer parameter updates during GRPO training."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from slime.backends.megatron_utils.actor import MegatronTrainRayActor

from .config import write_json


class AuditedActor(MegatronTrainRayActor):
    def train_actor(self, rollout_id, rollout_data, external_data=None):
        def parameters(optimizer):
            children = getattr(optimizer, "chained_optimizers", None)
            if children:
                for child in children:
                    yield from parameters(child)
                return
            inner = getattr(optimizer, "optimizer", optimizer)
            for group in getattr(inner, "param_groups", []):
                for parameter in group["params"]:
                    if isinstance(parameter, torch.Tensor) and parameter.numel():
                        yield parameter

        probes = []
        with torch.no_grad():
            for index, parameter in enumerate(parameters(self.optimizer)):
                if index >= 256:
                    break
                flat = parameter.detach().reshape(-1)
                stride = max(1, flat.numel() // 256)
                probes.append((parameter, stride, flat[::stride][:256].float().cpu().clone()))
        result = super().train_actor(rollout_id, rollout_data, external_data=external_data)
        changed = total = 0
        maximum = 0.0
        with torch.no_grad():
            for parameter, stride, before in probes:
                after = parameter.detach().reshape(-1)[::stride][:256].float().cpu()
                delta = (after - before).abs()
                if not bool(torch.isfinite(delta).all()):
                    raise RuntimeError("Non-finite optimizer parameter update")
                changed += int(torch.count_nonzero(delta).item())
                total += delta.numel()
                maximum = max(maximum, float(delta.max().item()))
        rank = torch.distributed.get_rank()
        root = Path(os.environ["R2E_COEVOLVE_ROOT"]) / "steps" / f"{rollout_id + 1:04d}"
        write_json(
            root / f"parameter_update_rank_{rank}.json",
            {
                "rank": rank,
                "step": rollout_id + 1,
                "probe_values": total,
                "changed_values": changed,
                "max_abs_delta": maximum,
                "kind": "bounded_optimizer_parameter_probes",
            },
        )
        return result
