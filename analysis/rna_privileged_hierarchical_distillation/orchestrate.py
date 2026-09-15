#!/usr/bin/env python3
"""Conservative vllab GPU dispatcher for the frozen RPH experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS = (17, 29, 43)
FOLDS = tuple(range(5))
CONDITIONS = ("route", "within", "align", "route_align", "route_within", "full",
              "full_shuffled", "strong_lineage")


def complete(root: Path, action: str, condition: str, seed: int, fold: int) -> bool:
    name = "selection.json" if action == "select" else (
        "refit_teacher.pt" if condition == "teacher" else "predictions.npz")
    return (root / condition / f"seed{seed}" / f"fold{fold}" / name).exists()


def stages(root: Path):
    yield "teacher_selection", [("select", "teacher", 17, fold) for fold in FOLDS]
    for condition in CONDITIONS:
        yield f"{condition}_selection", [("select", condition, 17, fold) for fold in FOLDS]
    # No new outer-test artifacts are created until every student choice is frozen.
    yield "teacher_refit", [("refit", "teacher", seed, fold) for seed in SEEDS for fold in FOLDS]
    for condition in CONDITIONS:
        yield f"{condition}_refit", [("refit", condition, seed, fold) for seed in SEEDS for fold in FOLDS]


def gpu_snapshot(requested: set[int]) -> dict[int, tuple[int, int]]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in output.splitlines():
        gpu, memory, utilization = (int(value.strip()) for value in line.split(","))
        if gpu in requested: result[gpu] = (memory, utilization)
    return result


def command(args, task):
    action, condition, seed, fold = task
    return [args.python, str(args.runner), action, "--condition", condition, "--seed", str(seed),
            "--fold", str(fold), "--output", str(args.output), "--apt-cache", str(args.apt_cache),
            "--rna-root", str(args.rna_root), "--device", "cuda"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--apt-cache", required=True, type=Path)
    parser.add_argument("--rna-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run.py"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-memory-mib", type=int, default=2000)
    parser.add_argument("--idle-utilization", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock_file = (args.output / "orchestrator.lock").open("w")
    try: fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("Another RPH orchestrator already holds the lock")
    gate = json.loads((args.output / "baseline_gate.json").read_text())
    if gate.get("status") != "PASS": raise SystemExit("Baseline gate has not passed")
    requested = {int(value) for value in args.gpus.split(",")}
    logs = args.output / "logs"; logs.mkdir(exist_ok=True)
    idle_streak = {gpu: 0 for gpu in requested}; last_wait_message = 0.0
    for stage, tasks in stages(args.output):
        pending = [task for task in tasks if not complete(args.output, *task)]
        while pending:
            if not (args.rna_root / "manifest.json").exists():
                if time.time() - last_wait_message > 600:
                    print("WAIT RNA manifest", args.rna_root, flush=True); last_wait_message = time.time()
                time.sleep(args.poll_seconds); continue
            snapshot = gpu_snapshot(requested)
            free = []
            for gpu in sorted(requested):
                memory, utilization = snapshot[gpu]
                idle_streak[gpu] = idle_streak[gpu] + 1 if (
                    memory < args.idle_memory_mib and utilization < args.idle_utilization) else 0
                if idle_streak[gpu] >= 2: free.append(gpu)
            if not free:
                if time.time() - last_wait_message > 600:
                    print("WAIT free GPU", snapshot, flush=True); last_wait_message = time.time()
                time.sleep(args.poll_seconds); continue
            wave = pending[:len(free)]; processes = []
            print("DISPATCH", stage, wave, "GPUS", free[:len(wave)], flush=True)
            for gpu, task in zip(free, wave):
                action, condition, seed, fold = task
                path = logs / f"{action}_{condition}_seed{seed}_fold{fold}.log"
                stream = path.open("a")
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                processes.append((task, stream, subprocess.Popen(command(args, task), stdout=stream,
                    stderr=subprocess.STDOUT, env=environment)))
                idle_streak[gpu] = 0
            failed = []
            for task, stream, process in processes:
                code = process.wait(); stream.close()
                if code != 0: failed.append((task, code))
            if failed:
                (args.output / "orchestrator_failure.json").write_text(json.dumps({
                    "stage": stage, "failed": failed, "time": time.time()}, indent=2) + "\n")
                raise SystemExit(f"Fail-closed at {stage}: {failed}")
            pending = [task for task in tasks if not complete(args.output, *task)]
        print("STAGE COMPLETE", stage, flush=True)
    (args.output / "training_complete.json").write_text(json.dumps({
        "status": "PASS", "conditions": list(CONDITIONS), "seeds": list(SEEDS),
        "folds": list(FOLDS), "time": time.time()}, indent=2) + "\n")
    print("ALL TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
