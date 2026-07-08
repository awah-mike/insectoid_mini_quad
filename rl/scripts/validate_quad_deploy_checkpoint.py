#!/usr/bin/env python3
"""Validate a 60D/12D quad deploy checkpoint without launching Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--obs-dim", type=int, default=60)
parser.add_argument("--action-dim", type=int, default=12)
parser.add_argument("--samples", type=int, default=8)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()


def build_actor(obs_dim: int, action_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, action_dim),
    )


def main() -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint does not contain model_state_dict")

    actor = build_actor(args.obs_dim, args.action_dim)
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in state_dict.items()
        if key.startswith("actor.")
    }
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"actor state mismatch: missing={missing}, unexpected={unexpected}")
    actor.eval()

    observations = torch.zeros(args.samples, args.obs_dim)
    observations[:, 6:9] = torch.tensor([0.0, 0.0, -1.0])
    with torch.inference_mode():
        actions = actor(observations)
    if actions.shape != (args.samples, args.action_dim):
        raise RuntimeError(f"bad action shape {actions.shape}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("non-finite action output")

    clipped = torch.clamp(actions, -1.0, 1.0)
    result = {
        "checkpoint": str(args.checkpoint),
        "iter": int(checkpoint.get("iter", -1)),
        "obs_dim": args.obs_dim,
        "action_dim": args.action_dim,
        "action_mean_abs": float(actions.abs().mean().item()),
        "action_max_abs": float(actions.abs().max().item()),
        "clipped_action_mean_abs": float(clipped.abs().mean().item()),
        "clipped_action_max_abs": float(clipped.abs().max().item()),
        "infos_keys": sorted((checkpoint.get("infos") or {}).keys()),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
