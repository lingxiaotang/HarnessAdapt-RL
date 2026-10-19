"""Controller for tool and hook evolution across Epoch 1 (H0 -> H1) and Epoch 2 (H1 -> H2)."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from .common import BASE, IMAGE, PACKAGE_ROOT, WORKSPACE, decision, files, read, sha, write
from harnessadapt.training.resource_guard import docker


def container(root: Path, name: str, gpu: bool = False) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "host",
        "--ipc",
        "host",
        "--label",
        f"self_evolve.coevolve_root={root}",
        "--memory",
        "256g",
        "-v",
        f"{WORKSPACE}:{WORKSPACE}",
        "-v",
        f"{PACKAGE_ROOT}:{PACKAGE_ROOT}",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "-w",
        str(WORKSPACE),
        "-e",
        f"PYTHONPATH=/root/Megatron-LM:{WORKSPACE}/slime:{PACKAGE_ROOT}/src:{WORKSPACE}/R2E-Gym/src",
    ]
    if gpu:
        command += [
            "--runtime",
            "nvidia",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=0,1,2,3",
            "--shm-size",
            "64g",
        ]
    return command + [IMAGE]


def fetch(port: int, route: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/{route}", timeout=3) as response:
        return json.load(response)


class Controller:
    def __init__(self, root: Path):
        self.root = root
        self.config = read(root / "config.json")
        self.status = read(root / "status.json")
        self.server = None
        self.server_name = "evolution-server-" + root.name[-24:]

    def save(self, **changes) -> None:
        if changes.get("stage") != "failed":
            self.status.pop("error", None)
        self.status.update(changes, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(self.root / "status.json", self.status)
        print(json.dumps(changes, ensure_ascii=False), flush=True)

    def start_server(self, model: Path) -> None:
        self.save(stage="starting_server", server_model=str(model))
        command = container(self.root, self.server_name, gpu=True) + [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.config["server_port"]),
            "--tp",
            str(self.config["server_tp"]),
            "--context-length",
            str(self.config["context_tokens"]),
            "--mem-fraction-static",
            "0.72",
            "--max-running-requests",
            str(self.config["server_max_requests"]),
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
            "--trust-remote-code",
        ]
        write(self.root / "server_command.json", command)
        self.server_log = (self.root / ("server_" + self.status["branch"] + ".log")).open("a", encoding="utf-8")
        self.server = subprocess.Popen(command, stdout=self.server_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError("Inference server exited; inspect server log")
            try:
                info = fetch(self.config["server_port"], "get_server_info")
                write(
                    self.root / "server.json",
                    dict(
                        model=str(model),
                        server_info=info,
                        model_config_sha256=sha(Path(model) / "config.json"),
                    ),
                )
                return
            except (OSError, ValueError):
                time.sleep(5)
        raise TimeoutError("Inference server did not become ready")

    def batch(self, index: int) -> Path:
        ids = read(self.root / "sampling_plan.json")["batches"][index]
        path = self.root / "batches" / f"{index:04d}.jsonl"
        if not path.exists():
            selected = {}
            with (self.root / "pool.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["label"] in ids:
                        selected[row["label"]] = row
            path.parent.mkdir(exist_ok=True)
            path.write_text("".join(json.dumps(selected[key]) + "\n" for key in ids), encoding="utf-8")
        return path

    def evaluate(self, output: Path, batch: Path, harness: Path | None) -> dict:
        if (output / "summary.json").exists() and read(output / "summary.json")["complete"]:
            return read(output / "summary.json")
        output.mkdir(parents=True, exist_ok=True)
        command = container(self.root, "evolution-rollout-" + str(os.getpid())) + [
            "python",
            "-u",
            "-m",
            "harnessadapt.evolution.evaluate",
            "--run-dir",
            str(self.root),
            "--output",
            str(output),
            "--input",
            str(batch),
        ]
        if harness:
            command += ["--harness", str(harness)]
        write(output / "command.json", command)
        self.save(stage="sampling", current_output=str(output), current_harness=str(harness) if harness else None)
        with (output / "rollout.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Rollout failed/incomplete ({result.returncode}): {output}")
        return read(output / "summary.json")

    def main(self) -> None:
        branch = self.status["branch"]
        model = BASE if branch == "h1" else Path(self.config["epoch1_checkpoint"])
        self.start_server(model)
        baseline = self.root / "h1" / "baseline"
        from .analyze import analyze
        from .audit_results import verify_results

        if not self.status["rounds"] and branch == "h1":
            self.evaluate(baseline, self.batch(0), None)
            verify_results(baseline)
            analyze(baseline)
            self.save(
                stage="awaiting_proposal",
                evidence=str(baseline / "analysis.json"),
                current_output=str(baseline),
            )
        elif branch == "h2" and not any(read(Path(p))["proposal"]["branch"] == "h2" for p in self.status["rounds"]):
            h2_baseline = self.root / "h2/baseline"
            self.evaluate(h2_baseline, self.batch(23), Path(self.status["incumbent"]))
            verify_results(h2_baseline)
            analyze(h2_baseline)
            self.save(
                stage="awaiting_proposal",
                evidence=str(h2_baseline / "analysis.json"),
                current_output=str(h2_baseline),
            )
        else:
            self.save(stage="awaiting_proposal")

        while True:
            requests = sorted((self.root / "requests").glob("*.json"))
            pending = [
                p
                for p in requests
                if not p.with_suffix(".result.json").exists() and not p.name.endswith(".result.json")
            ]
            if not pending:
                if self.server.poll() is not None:
                    raise RuntimeError("Inference server died while waiting for proposal")
                time.sleep(3)
                continue
            path = pending[0]
            request = read(path)
            if request["action"] == "candidate":
                branch = self.status["branch"]
                parent = Path(self.status["incumbent"]) if self.status["incumbent"] else None
                candidate = Path(request["harness"]).resolve()
                index = int(request["batch_index"])
                batch = self.batch(index)
                round_dir = self.root / branch / "rounds" / path.stem
                parent_output = baseline if branch == "h1" and parent is None and index == 0 else round_dir / "parent"
                p = self.evaluate(parent_output, batch, parent)
                c = self.evaluate(round_dir / "candidate", batch, candidate)
                from .report import report, verify_pair

                verify_pair(parent_output, round_dir / "candidate")
                verify_results(parent_output)
                verify_results(round_dir / "candidate")
                analyze(round_dir / "candidate")
                result = decision(p, c)
                result.update(
                    proposal=request,
                    parent_harness=str(parent) if parent else None,
                    candidate_harness=str(candidate),
                    candidate_files=files(candidate),
                )
                write(round_dir / "decision.json", result)
                write(path.with_suffix(".result.json"), result)
                self.status["rounds"].append(str(round_dir / "decision.json"))
                self.status["accepted" if result["accepted"] else "rejected"].append(str(round_dir))
                if result["accepted"]:
                    self.status["incumbent"] = str(candidate)
                self.save(
                    stage="awaiting_proposal",
                    last_decision=result["status"],
                    current_output=str(round_dir / "candidate"),
                    evidence=str(round_dir / "candidate" / "analysis.json"),
                )
                report(self.root)
            elif request["action"] == "finish_branch":
                branch = self.status["branch"]
                incumbent = self.status.get("incumbent")
                write(
                    self.root / branch / "endpoint_verified.json",
                    dict(
                        incumbent=incumbent,
                        target_files=files(Path(incumbent)),
                        accepted_rounds=self.status["accepted"],
                        source_model=str(model),
                    ),
                )
                if branch == "h1":
                    docker("stop", "--time", "5", self.server_name, timeout=15)
                    self.server.wait(timeout=15)
                    self.save(stage="starting_epoch2", branch="h2", incumbent=incumbent)
                    model = Path(self.config["epoch1_checkpoint"])
                    self.start_server(model)
                    h2_baseline = self.root / "h2/baseline"
                    self.evaluate(h2_baseline, self.batch(23), Path(incumbent))
                    verify_results(h2_baseline)
                    analyze(h2_baseline)
                    self.save(
                        stage="awaiting_proposal",
                        evidence=str(h2_baseline / "analysis.json"),
                        current_output=str(h2_baseline),
                    )
                    write(path.with_suffix(".result.json"), dict(status="h1_verified_h2_started"))
                else:
                    from .report import report

                    report(self.root)
                    write(path.with_suffix(".result.json"), dict(status="h2_endpoint_verified"))
                    self.save(stage="complete")
                    return


def main(root: Path) -> None:
    root = root.resolve()
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ctl = Controller(root)
    try:
        ctl.main()
    except BaseException as exc:
        ctl.save(stage="failed", error=repr(exc))
        raise
    finally:
        try:
            docker("stop", "--time", "3", ctl.server_name, timeout=10)
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    main(p.parse_args().run_dir)
