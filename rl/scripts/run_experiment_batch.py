#!/usr/bin/env python3
"""Run a batch of training experiments and score each with the strict evaluator.

This driver automates the manual train -> evaluate -> log loop used throughout
the quad gait work. For each (experiment, seed) pair in a batch spec it:

1. launches Isaac Lab RSL-RL training via rl/scripts/train_rsl_rl.py,
2. locates the produced run directory and its final checkpoint,
3. runs rl/scripts/evaluate_rsl_rl.py (strict evaluator) on that checkpoint,
4. optionally renders a review video,
5. appends a scored row to rl/experiments/EXPERIMENTS.tsv.

The driver itself is plain Python (no Isaac imports); every sim stage is a
subprocess through isaaclab.sh. Run it directly with python3, ideally in the
background for multi-hour batches.

Example:

    python3 rl/scripts/run_experiment_batch.py \
        --spec rl/experiments/quick_wins_gait_refine.json

Spec format (JSON):

    {
      "batch_name": "quickwin",
      "defaults": {
        "task": "Isaac-InsectoidMiniQuad-GaitRefine-Direct-v0",
        "eval_task": "Isaac-InsectoidMiniQuad-GaitRefine-Direct-Play-v0",
        "num_envs": 4096,
        "max_iterations": 150,
        "seeds": [42, 123],
        "eval_steps": 500,
        "eval_num_envs": 16,
        "eval_forward_vel": 0.24,
        "resume": null,
        "render": false
      },
      "experiments": [
        {"id": "baseline", "overrides": []},
        {"id": "obsnorm", "overrides": ["agent.policy.actor_obs_normalization=True"]}
      ]
    }

Per-experiment keys override the defaults. "overrides" entries are Hydra
overrides forwarded verbatim to train.py (env.* / agent.*). "resume" is
{"load_run": ..., "checkpoint": ...}; the checkpoint may be an absolute path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "rl/experiments/EXPERIMENTS.tsv"
HARNESS_LOG_DIR = REPO_ROOT / "outputs/harness_logs"

LEDGER_COLUMNS = [
    "date_utc",
    "batch",
    "experiment_id",
    "seed",
    "task",
    "run_dir",
    "checkpoint",
    "task_score",
    "mean_forward_vel_y_mps",
    "mean_abs_yaw_rate_radps",
    "mean_latest_touchdown_stride_m",
    "mean_touchdown_rate_hz",
    "contact_fraction_imbalance",
    "score_tracking",
    "score_posture",
    "score_gait",
    "resets",
    "eval_file",
    "video",
    "overrides",
    "status",
    "notes",
]


def resolve_isaaclab_sh() -> Path:
    candidates = []
    if os.environ.get("ISAACLAB_PATH"):
        candidates.append(Path(os.environ["ISAACLAB_PATH"]).expanduser())
    candidates.extend([Path("/home/ubuntu/IsaacLab"), Path("/workspace/isaaclab"), Path("/opt/IsaacLab")])
    for root in candidates:
        launcher = root / "isaaclab.sh"
        if launcher.is_file():
            return launcher
    raise FileNotFoundError(f"Could not find isaaclab.sh in: {[str(c) for c in candidates]}")


def run_stage(cmd: list[str], log_file: Path, dry_run: bool) -> int:
    print(f"[harness] $ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TERM="xterm")
    with log_file.open("a") as handle:
        handle.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n$ {' '.join(cmd)}\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return result.returncode


def find_run_dir(run_name: str) -> Path | None:
    matches = sorted(REPO_ROOT.glob(f"logs/rsl_rl/*/????-??-??_??-??-??_{run_name}"))
    return matches[-1] if matches else None


def find_final_checkpoint(run_dir: Path) -> Path | None:
    best_iter, best_path = -1, None
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and int(match.group(1)) > best_iter:
            best_iter, best_path = int(match.group(1)), path
    return best_path


def parse_eval_file(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            metrics[key.strip()] = value.strip()
    return metrics


def append_ledger_row(row: dict[str, str]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LEDGER_PATH.exists()
    with LEDGER_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, delimiter="\t", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "n/a") for key in LEDGER_COLUMNS})


def run_experiment(exp: dict, defaults: dict, batch_name: str, launcher: Path, seed: int, args) -> dict:
    cfg = {**defaults, **exp}
    exp_id = cfg["id"]
    run_name = f"{batch_name}_{exp_id}_s{seed}"
    log_file = HARNESS_LOG_DIR / f"{run_name}.log"
    row = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "batch": batch_name,
        "experiment_id": exp_id,
        "seed": str(seed),
        "task": cfg["task"],
        "overrides": " ".join(cfg.get("overrides", [])) or "none",
        "notes": cfg.get("notes", ""),
        "status": "failed",
    }

    train_cmd = [
        str(launcher), "-p", "rl/scripts/train_rsl_rl.py",
        "--task", cfg["task"],
        "--headless",
        "--num_envs", str(cfg["num_envs"]),
        "--max_iterations", str(cfg["max_iterations"]),
        "--seed", str(seed),
        "--run_name", run_name,
    ]
    resume = cfg.get("resume")
    if resume:
        train_cmd += ["--resume", "--load_run", resume["load_run"], "--checkpoint", resume["checkpoint"]]
    train_cmd += cfg.get("overrides", [])

    if run_stage(train_cmd, log_file, args.dry_run) != 0:
        row["notes"] = f"training failed; see {log_file}"
        return row
    if args.dry_run:
        row["status"] = "dry_run"
        return row

    run_dir = find_run_dir(run_name)
    checkpoint = find_final_checkpoint(run_dir) if run_dir else None
    if checkpoint is None:
        row["notes"] = f"no checkpoint found for run {run_name}; see {log_file}"
        return row
    row["run_dir"] = str(run_dir.relative_to(REPO_ROOT))
    row["checkpoint"] = checkpoint.name

    eval_file = run_dir / f"eval_harness_{checkpoint.stem}.txt"
    eval_cmd = [
        str(launcher), "-p", "rl/scripts/evaluate_rsl_rl.py",
        "--task", cfg["eval_task"],
        "--checkpoint", str(checkpoint),
        "--num_envs", str(cfg["eval_num_envs"]),
        "--steps", str(cfg["eval_steps"]),
        "--forward-vel", str(cfg["eval_forward_vel"]),
        "--output", str(eval_file),
    ]
    if run_stage(eval_cmd, log_file, args.dry_run) != 0 or not eval_file.exists():
        row["notes"] = f"evaluation failed; see {log_file}"
        return row
    metrics = parse_eval_file(eval_file)
    row.update(
        {
            "task_score": metrics.get("TASK_SCORE", "n/a"),
            "mean_forward_vel_y_mps": metrics.get("MEAN_FORWARD_VEL_Y_MPS", "n/a"),
            "mean_abs_yaw_rate_radps": metrics.get("MEAN_ABS_YAW_RATE_RADPS", "n/a"),
            "mean_latest_touchdown_stride_m": metrics.get("MEAN_LATEST_TOUCHDOWN_STRIDE_M", "n/a"),
            "mean_touchdown_rate_hz": metrics.get("MEAN_TOUCHDOWN_RATE_HZ", "n/a"),
            "contact_fraction_imbalance": metrics.get("CONTACT_FRACTION_IMBALANCE", "n/a"),
            "score_tracking": metrics.get("SCORE_TRACKING", "n/a"),
            "score_posture": metrics.get("SCORE_POSTURE", "n/a"),
            "score_gait": metrics.get("SCORE_GAIT", "n/a"),
            "resets": metrics.get("RESETS", "n/a"),
            "eval_file": str(eval_file.relative_to(REPO_ROOT)),
        }
    )

    if cfg.get("render"):
        video_path = run_dir / "videos" / f"harness_{checkpoint.stem}.mp4"
        render_cmd = [
            str(launcher), "-p", "rl/scripts/render_policy_video.py",
            "--task", cfg["eval_task"],
            "--checkpoint", str(checkpoint),
            "--output", str(video_path),
            "--forward-vel", str(cfg["eval_forward_vel"]),
            "--video-length", str(cfg.get("render_length", 400)),
            "--headless",
        ]
        if run_stage(render_cmd, log_file, args.dry_run) == 0 and video_path.exists():
            row["video"] = str(video_path.relative_to(REPO_ROOT))
        else:
            row["video"] = "render_failed"

    row["status"] = "ok"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True, help="Batch spec JSON file.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiment ids.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    defaults = spec.get("defaults", {})
    batch_name = spec.get("batch_name", args.spec.stem)
    launcher = resolve_isaaclab_sh()

    experiments = spec["experiments"]
    if args.only:
        experiments = [exp for exp in experiments if exp["id"] in set(args.only)]
        if not experiments:
            print(f"[harness] no experiments matched --only {args.only}", file=sys.stderr)
            return 1

    total = sum(len(({**defaults, **exp}).get("seeds", [42])) for exp in experiments)
    print(f"[harness] batch '{batch_name}': {len(experiments)} experiments, {total} runs total", flush=True)

    start = time.time()
    failures = 0
    for exp in experiments:
        for seed in ({**defaults, **exp}).get("seeds", [42]):
            run_start = time.time()
            row = run_experiment(exp, defaults, batch_name, launcher, seed, args)
            if not args.dry_run:
                append_ledger_row(row)
            failures += row["status"] == "failed"
            print(
                f"[harness] {row['experiment_id']} seed={seed}: {row['status']}"
                f" task_score={row.get('task_score', 'n/a')}"
                f" ({(time.time() - run_start) / 60:.1f} min)",
                flush=True,
            )
    print(f"[harness] batch done in {(time.time() - start) / 60:.1f} min, {failures} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
