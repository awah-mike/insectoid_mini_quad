#!/usr/bin/env python3
"""Bake the BL tibia posture cleanup into a trained RSL-RL policy.

This script performs supervised actor fine-tuning from an existing checkpoint.
It collects corrected targets from the source policy, changes only the
BL_tibia action target, and trains the actor to emit that corrected action
directly. The saved checkpoint therefore does not require rollout-time action
clamping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "logs/rsl_rl/insectoid_mini_quad_gait_refine"
    / "2026-06-25_22-56-37_gait_refine_long_stride_v1_from_model_1399"
    / "model_1798.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "logs/rsl_rl/insectoid_mini_quad_bl_tibia_baked"
    / "model_1798_bl_tibia_baked_v1.pt"
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-GaitRefine-Direct-Play-v0")
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1800)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--minibatch-size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2.0e-4)
parser.add_argument("--forward-vel", type=float, default=0.30)
parser.add_argument("--post-bl-coxa-max-rad", type=float, default=None)
parser.add_argument("--post-bl-coxa-br-offset-rad", type=float, default=None)
parser.add_argument("--post-bl-coxa-br-target-offset-rad", type=float, default=None)
parser.add_argument("--post-bl-femur-max-rad", type=float, default=None)
parser.add_argument("--post-bl-femur-br-offset-rad", type=float, default=None)
parser.add_argument("--post-bl-femur-br-target-offset-rad", type=float, default=None)
parser.add_argument("--post-bl-tibia-max-rad", type=float, default=2.00)
parser.add_argument("--post-bl-tibia-br-offset-rad", type=float, default=0.30)
parser.add_argument("--post-bl-tibia-br-target-offset-rad", type=float, default=None)
parser.add_argument("--post-bl-correction-blend", type=float, default=1.0)
parser.add_argument(
    "--rollout-action-source",
    choices=("target", "teacher"),
    default="target",
    help="Use corrected target actions or original teacher actions while collecting the supervised dataset.",
)
parser.add_argument(
    "--correction-gate",
    choices=("all", "bl_swing", "bl_lifted"),
    default="all",
    help="Apply the BL correction on all samples, BL swing samples, or only lifted BL swing samples.",
)
parser.add_argument("--bl-lifted-min-height", type=float, default=0.025)
parser.add_argument("--mirror-bl-from-delayed-br", action="store_true")
parser.add_argument("--mirror-bl-joints", choices=("distal", "all"), default="distal")
parser.add_argument("--mirror-delay-steps", type=float, default=0.0)
parser.add_argument("--mirror-bl-blend", type=float, default=0.35)
parser.add_argument("--bl-loss-weight", type=float, default=8.0)
parser.add_argument("--preserve-loss-weight", type=float, default=1.0)
parser.add_argument("--log-interval", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def correct_bl_tibia_actions(
    actions: torch.Tensor,
    default_joint_pos: torch.Tensor,
    action_scale: float,
    bl_coxa_index: int,
    br_coxa_index: int,
    bl_femur_index: int,
    br_femur_index: int,
    bl_tibia_index: int,
    br_tibia_index: int,
    bl_coxa_max_rad: float | None,
    bl_coxa_br_offset_rad: float | None,
    bl_coxa_br_target_offset_rad: float | None,
    bl_femur_max_rad: float | None,
    bl_femur_br_offset_rad: float | None,
    bl_femur_br_target_offset_rad: float | None,
    bl_tibia_max_rad: float | None,
    bl_tibia_br_offset_rad: float | None,
    bl_tibia_br_target_offset_rad: float | None,
    blend: float | torch.Tensor,
) -> torch.Tensor:
    """Return action targets with BL rear-leg inward curl corrected."""
    target = default_joint_pos + action_scale * actions

    def adjust_joint(
        bl_index: int,
        br_index: int,
        max_rad: float | None,
        br_offset_rad: float | None,
        br_target_offset_rad: float | None,
    ) -> None:
        if max_rad is None and br_offset_rad is None and br_target_offset_rad is None:
            return
        corrected = target[:, bl_index]
        if max_rad is not None or br_offset_rad is not None:
            bl_limit = torch.full_like(corrected, float("inf"))
            if max_rad is not None:
                bl_limit = torch.minimum(bl_limit, torch.full_like(bl_limit, max_rad))
            if br_offset_rad is not None:
                bl_limit = torch.minimum(bl_limit, target[:, br_index] + br_offset_rad)
            corrected = torch.minimum(corrected, bl_limit)
        if br_target_offset_rad is not None:
            corrected = target[:, br_index] + br_target_offset_rad
        target[:, bl_index] = (1.0 - blend) * target[:, bl_index] + blend * corrected

    adjust_joint(
        bl_coxa_index,
        br_coxa_index,
        bl_coxa_max_rad,
        bl_coxa_br_offset_rad,
        bl_coxa_br_target_offset_rad,
    )
    adjust_joint(
        bl_femur_index,
        br_femur_index,
        bl_femur_max_rad,
        bl_femur_br_offset_rad,
        bl_femur_br_target_offset_rad,
    )
    adjust_joint(
        bl_tibia_index,
        br_tibia_index,
        bl_tibia_max_rad,
        bl_tibia_br_offset_rad,
        bl_tibia_br_target_offset_rad,
    )
    return torch.clamp((target - default_joint_pos) / action_scale, -1.0, 1.0)


def main() -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.lin_vel_x_range = (0.0, 0.0)
    env_cfg.lin_vel_y_range = (args.forward_vel, args.forward_vel)
    env_cfg.ang_vel_z_range = (0.0, 0.0)
    env_cfg.rel_standing_envs = 0.0
    if hasattr(env_cfg, "events"):
        env_cfg.events.physics_material = None

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[BAKE] loading source checkpoint: {args.checkpoint}", flush=True)
    runner.load(str(args.checkpoint), load_optimizer=False, map_location="cpu")
    print("[BAKE] source checkpoint loaded", flush=True)

    student = runner.alg.policy
    student.eval()
    print("[BAKE] student policy ready", flush=True)

    raw = raw_env.unwrapped
    print("[BAKE] raw env ready", flush=True)
    joint_names = tuple(raw._robot.data.joint_names)
    bl_coxa_index = joint_names.index("BL_coxa_joint")
    br_coxa_index = joint_names.index("BR_coxa_joint")
    bl_femur_index = joint_names.index("BL_femur_joint")
    br_femur_index = joint_names.index("BR_femur_joint")
    bl_tibia_index = joint_names.index("BL_tibia_joint")
    br_tibia_index = joint_names.index("BR_tibia_joint")
    if args.mirror_bl_joints == "all":
        mirror_bl_indices = (bl_coxa_index, bl_femur_index, bl_tibia_index)
        mirror_br_indices = (br_coxa_index, br_femur_index, br_tibia_index)
    else:
        mirror_bl_indices = (bl_femur_index, bl_tibia_index)
        mirror_br_indices = (br_femur_index, br_tibia_index)
    default_joint_pos = raw._robot.data.default_joint_pos
    num_actions = int(default_joint_pos.shape[-1])
    action_scale = float(raw.cfg.action_scale)
    print(
        f"[BAKE] joint indices"
        f" BL_coxa={bl_coxa_index} BR_coxa={br_coxa_index}"
        f" BL_femur={bl_femur_index} BR_femur={br_femur_index}"
        f" BL_tibia={bl_tibia_index} BR_tibia={br_tibia_index}"
        f" action_scale={action_scale:.3f} num_actions={num_actions}",
        flush=True,
    )
    blend = max(0.0, min(1.0, args.post_bl_correction_blend))
    mirror_blend = max(0.0, min(1.0, args.mirror_bl_blend))
    mirror_delay_steps = max(0.0, args.mirror_delay_steps)
    br_action_history: list[torch.Tensor] = []
    correction_gate_sum = 0.0
    correction_gate_count = 0

    def delayed_br_action(source: torch.Tensor) -> torch.Tensor:
        br_action_history.append(source.detach().clone())
        max_history = int(torch.ceil(torch.tensor(mirror_delay_steps)).item()) + 3
        if len(br_action_history) > max_history:
            del br_action_history[:-max_history]
        if mirror_delay_steps <= 0.0 or len(br_action_history) == 1:
            return source

        lower = int(torch.floor(torch.tensor(mirror_delay_steps)).item())
        upper = int(torch.ceil(torch.tensor(mirror_delay_steps)).item())
        frac = mirror_delay_steps - lower

        def history_at_lag(lag: int) -> torch.Tensor:
            index = max(0, len(br_action_history) - 1 - lag)
            return br_action_history[index]

        newer = history_at_lag(lower)
        older = history_at_lag(upper)
        return (1.0 - frac) * newer + frac * older

    raw.episode_length_buf.zero_()
    raw._commands[:, 0] = 0.0
    raw._commands[:, 1] = args.forward_vel
    raw._commands[:, 2] = 0.0
    obs = env.get_observations().to(args.device)
    print("[BAKE] initial observations ready", flush=True)

    with torch.no_grad():
        initial_actor_obs = student.actor_obs_normalizer(student.get_actor_obs(obs))
    print(f"[BAKE] actor obs dim={initial_actor_obs.shape[-1]}", flush=True)
    sample_count = args.steps * args.num_envs
    dataset_obs = torch.empty(
        (sample_count, initial_actor_obs.shape[-1]),
        dtype=initial_actor_obs.dtype,
        device=args.device,
    )
    dataset_actions = torch.empty(
        (sample_count, num_actions),
        dtype=torch.float32,
        device=args.device,
    )
    print(f"[BAKE] collecting {args.steps * args.num_envs} corrected-action samples", flush=True)
    for step in range(args.steps):
        with torch.no_grad():
            raw._commands[:, 0] = 0.0
            raw._commands[:, 1] = args.forward_vel
            raw._commands[:, 2] = 0.0
            teacher_actions = student.act_inference(obs)
            teacher_target_actions = teacher_actions
            if args.mirror_bl_from_delayed_br:
                teacher_target_actions = teacher_actions.clone()
                delayed_br = delayed_br_action(teacher_actions[:, list(mirror_br_indices)])
                teacher_target_actions[:, list(mirror_bl_indices)] = (
                    (1.0 - mirror_blend) * teacher_target_actions[:, list(mirror_bl_indices)]
                    + mirror_blend * delayed_br
                )
            correction_blend = blend
            if args.correction_gate != "all":
                foot_contacts = raw._get_foot_contacts()
                gate = (~foot_contacts[:, 0]).float()
                if args.correction_gate == "bl_lifted":
                    foot_pos_w = raw._robot.data.body_pos_w[:, raw._feet_body_ids, :]
                    foot_height = torch.clamp(
                        foot_pos_w[:, :, 2] - raw._terrain.env_origins[:, 2].unsqueeze(1),
                        min=0.0,
                    )
                    gate = gate * (foot_height[:, 0] > args.bl_lifted_min_height).float()
                correction_blend = blend * gate
                correction_gate_sum += float(gate.sum().detach().cpu())
                correction_gate_count += int(gate.numel())
            else:
                correction_gate_sum += float(args.num_envs)
                correction_gate_count += args.num_envs
            target_actions = correct_bl_tibia_actions(
                teacher_target_actions,
                default_joint_pos,
                action_scale,
                bl_coxa_index,
                br_coxa_index,
                bl_femur_index,
                br_femur_index,
                bl_tibia_index,
                br_tibia_index,
                args.post_bl_coxa_max_rad,
                args.post_bl_coxa_br_offset_rad,
                args.post_bl_coxa_br_target_offset_rad,
                args.post_bl_femur_max_rad,
                args.post_bl_femur_br_offset_rad,
                args.post_bl_femur_br_target_offset_rad,
                args.post_bl_tibia_max_rad,
                args.post_bl_tibia_br_offset_rad,
                args.post_bl_tibia_br_target_offset_rad,
                correction_blend,
            )
            actor_obs = student.actor_obs_normalizer(student.get_actor_obs(obs))
            start = step * args.num_envs
            end = start + args.num_envs
            dataset_obs[start:end].copy_(actor_obs)
            dataset_actions[start:end].copy_(target_actions)
            rollout_actions = teacher_actions if args.rollout_action_source == "teacher" else target_actions
            obs, _, dones, _ = env.step(rollout_actions)
            obs = obs.to(args.device)
            if bool(torch.any(dones).item()):
                raw.episode_length_buf[dones.to(raw.device) > 0] = 0

        if (step + 1) % args.log_interval == 0 or step == 0:
            with torch.no_grad():
                joint_pos = raw._robot.data.joint_pos
                bl_coxa_mean = joint_pos[:, bl_coxa_index].mean().item()
                br_coxa_mean = joint_pos[:, br_coxa_index].mean().item()
                bl_femur_mean = joint_pos[:, bl_femur_index].mean().item()
                br_femur_mean = joint_pos[:, br_femur_index].mean().item()
                bl_mean = joint_pos[:, bl_tibia_index].mean().item()
                br_mean = joint_pos[:, br_tibia_index].mean().item()
                offset_mean = (joint_pos[:, bl_tibia_index] - joint_pos[:, br_tibia_index]).mean().item()
                vel_y = raw._robot.data.root_lin_vel_b[:, 1].mean().item()
            print(
                "[BAKE][collect]"
                f" step={step + 1}/{args.steps}"
                f" coxa_bl_minus_br={bl_coxa_mean - br_coxa_mean:.3f}"
                f" femur_bl_minus_br={bl_femur_mean - br_femur_mean:.3f}"
                f" bl={bl_mean:.3f}"
                f" br={br_mean:.3f}"
                f" bl_minus_br={offset_mean:.3f}"
                f" vel_y={vel_y:.3f}",
                flush=True,
            )

    action_weights = torch.full(
        (num_actions,),
        args.preserve_loss_weight,
        dtype=torch.float32,
        device=args.device,
    )
    if (
        args.post_bl_coxa_max_rad is not None
        or args.post_bl_coxa_br_offset_rad is not None
        or args.post_bl_coxa_br_target_offset_rad is not None
    ):
        action_weights[bl_coxa_index] = args.bl_loss_weight
    if (
        args.post_bl_femur_max_rad is not None
        or args.post_bl_femur_br_offset_rad is not None
        or args.post_bl_femur_br_target_offset_rad is not None
    ):
        action_weights[bl_femur_index] = args.bl_loss_weight
    if (
        args.post_bl_tibia_max_rad is not None
        or args.post_bl_tibia_br_offset_rad is not None
        or args.post_bl_tibia_br_target_offset_rad is not None
    ):
        action_weights[bl_tibia_index] = args.bl_loss_weight
    if args.mirror_bl_from_delayed_br:
        for index in mirror_bl_indices:
            action_weights[index] = args.bl_loss_weight
    optimizer = torch.optim.Adam(student.actor.parameters(), lr=args.lr)
    student.train()

    sample_count = dataset_obs.shape[0]
    print(
        f"[BAKE] fitting actor on {sample_count} samples for {args.epochs} epochs"
        f" minibatch={args.minibatch_size}",
        flush=True,
    )
    for epoch in range(args.epochs):
        permutation = torch.randperm(sample_count, device=args.device)
        epoch_loss = 0.0
        epoch_bl_loss = 0.0
        batches = 0
        for start in range(0, sample_count, args.minibatch_size):
            ids = permutation[start : start + args.minibatch_size]
            pred_actions = student.actor(dataset_obs[ids])
            per_action_loss = (pred_actions - dataset_actions[ids]).square()
            loss = (per_action_loss * action_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_bl_loss += float(
                (
                    per_action_loss[:, bl_coxa_index]
                    + per_action_loss[:, bl_femur_index]
                    + per_action_loss[:, bl_tibia_index]
                ).mean().detach().cpu()
            )
            batches += 1
        print(
            "[BAKE][fit]"
            f" epoch={epoch + 1}/{args.epochs}"
            f" loss={epoch_loss / batches:.6f}"
            f" bl_mse={epoch_bl_loss / batches:.6f}",
            flush=True,
        )
    correction_gate_fraction = correction_gate_sum / max(1, correction_gate_count)

    student.eval()
    verify_bl_coxa_offsets = []
    verify_bl_femur_offsets = []
    verify_bl_tibia_offsets = []
    verify_vel_y = []
    print("[BAKE] verifying baked actor without action correction", flush=True)
    for step in range(360):
        with torch.no_grad():
            raw._commands[:, 0] = 0.0
            raw._commands[:, 1] = args.forward_vel
            raw._commands[:, 2] = 0.0
            actions = torch.clamp(student.act_inference(obs), -1.0, 1.0)
            obs, _, dones, _ = env.step(actions)
            obs = obs.to(args.device)
            if step >= 80:
                joint_pos = raw._robot.data.joint_pos
                verify_bl_coxa_offsets.append(
                    (joint_pos[:, bl_coxa_index] - joint_pos[:, br_coxa_index]).detach().cpu()
                )
                verify_bl_femur_offsets.append(
                    (joint_pos[:, bl_femur_index] - joint_pos[:, br_femur_index]).detach().cpu()
                )
                verify_bl_tibia_offsets.append(
                    (joint_pos[:, bl_tibia_index] - joint_pos[:, br_tibia_index]).detach().cpu()
                )
                verify_vel_y.append(raw._robot.data.root_lin_vel_b[:, 1].detach().cpu())
            if bool(torch.any(dones).item()):
                raw.episode_length_buf[dones.to(raw.device) > 0] = 0
    verify_bl_coxa_offset = torch.cat(verify_bl_coxa_offsets)
    verify_bl_femur_offset = torch.cat(verify_bl_femur_offsets)
    verify_bl_tibia_offset = torch.cat(verify_bl_tibia_offsets)
    verify_vel = torch.cat(verify_vel_y)
    print(
        "[BAKE][verify]"
        f" actual_bl_coxa_minus_br_mean={verify_bl_coxa_offset.mean().item():.4f}"
        f" actual_bl_femur_minus_br_mean={verify_bl_femur_offset.mean().item():.4f}"
        f" actual_bl_tibia_minus_br_mean={verify_bl_tibia_offset.mean().item():.4f}"
        f" actual_bl_tibia_minus_br_p95={verify_bl_tibia_offset.quantile(0.95).item():.4f}"
        f" mean_vel_y={verify_vel.mean().item():.4f}"
        f" vel_y_std={verify_vel.std(unbiased=False).item():.4f}",
        flush=True,
    )

    source = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    saved = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "iter": int(source.get("iter", 0)),
        "infos": {
            "source_checkpoint": str(args.checkpoint),
            "method": "supervised_actor_distillation_bl_tibia_target_correction",
            "task": args.task,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "epochs": args.epochs,
            "minibatch_size": args.minibatch_size,
            "lr": args.lr,
            "forward_vel": args.forward_vel,
            "post_bl_coxa_max_rad": args.post_bl_coxa_max_rad,
            "post_bl_coxa_br_offset_rad": args.post_bl_coxa_br_offset_rad,
            "post_bl_coxa_br_target_offset_rad": args.post_bl_coxa_br_target_offset_rad,
            "post_bl_femur_max_rad": args.post_bl_femur_max_rad,
            "post_bl_femur_br_offset_rad": args.post_bl_femur_br_offset_rad,
            "post_bl_femur_br_target_offset_rad": args.post_bl_femur_br_target_offset_rad,
            "post_bl_tibia_max_rad": args.post_bl_tibia_max_rad,
            "post_bl_tibia_br_offset_rad": args.post_bl_tibia_br_offset_rad,
            "post_bl_tibia_br_target_offset_rad": args.post_bl_tibia_br_target_offset_rad,
            "post_bl_correction_blend": blend,
            "rollout_action_source": args.rollout_action_source,
            "correction_gate": args.correction_gate,
            "bl_lifted_min_height": args.bl_lifted_min_height,
            "correction_gate_fraction": correction_gate_fraction,
            "mirror_bl_from_delayed_br": args.mirror_bl_from_delayed_br,
            "mirror_bl_joints": args.mirror_bl_joints,
            "mirror_delay_steps": args.mirror_delay_steps,
            "mirror_bl_blend": mirror_blend,
            "bl_loss_weight": args.bl_loss_weight,
            "preserve_loss_weight": args.preserve_loss_weight,
            "verify_actual_bl_coxa_minus_br_mean": verify_bl_coxa_offset.mean().item(),
            "verify_actual_bl_femur_minus_br_mean": verify_bl_femur_offset.mean().item(),
            "verify_actual_bl_tibia_minus_br_mean": verify_bl_tibia_offset.mean().item(),
            "verify_actual_bl_tibia_minus_br_p95": verify_bl_tibia_offset.quantile(0.95).item(),
            "verify_mean_vel_y": verify_vel.mean().item(),
            "verify_vel_y_std": verify_vel.std(unbiased=False).item(),
        },
    }
    torch.save(saved, args.output)

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(saved["infos"], indent=2) + "\n")
    env.close()
    print(f"CHECKPOINT={args.output}", flush=True)
    print(f"SUMMARY={summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
