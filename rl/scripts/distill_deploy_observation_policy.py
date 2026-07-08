#!/usr/bin/env python3
"""Distill an accepted policy onto deployment-style observations.

The source policy is treated as a frozen teacher. During collection the teacher
receives privileged simulator base linear velocity, while the student receives
the same 60D quad-test observation with deployment-safe base velocity
substitution and optional sensor noise. The actor is then fit to reproduce the
teacher actions from the deployment observation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-GaitRefineDeploy-Direct-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--minibatch-size", type=int, default=8192)
parser.add_argument("--lr", type=float, default=1.0e-4)
parser.add_argument("--forward-vel", type=float, default=0.30)
parser.add_argument("--command-jitter", type=float, default=0.02)
parser.add_argument(
    "--target-action-clip",
    type=float,
    default=1.0,
    help="Distill the effective teacher action after symmetric clipping. Use <=0 to disable.",
)
parser.add_argument(
    "--action-limit-loss-weight",
    type=float,
    default=0.05,
    help="Extra loss on student actions outside the target clip range.",
)
parser.add_argument(
    "--student-base-lin-vel-observation",
    choices=("kinematic", "zero"),
    default="kinematic",
    help="Deployment-safe obs[0:3] source used by the student.",
)
parser.add_argument(
    "--student-noise-samples",
    type=int,
    default=2,
    help="Noisy student observations collected per simulated state.",
)
parser.add_argument(
    "--disable-student-observation-noise",
    action="store_true",
    help="Collect clean deployment observations instead of noisy ones.",
)
parser.add_argument(
    "--rollout-action-source",
    choices=("teacher", "student"),
    default="teacher",
    help="Which policy drives the simulator while collecting the dataset.",
)
parser.add_argument("--verify-steps", type=int, default=420)
parser.add_argument("--log-interval", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _set_base_velocity_mode(raw_env_or_cfg, mode: str) -> None:
    cfg = getattr(raw_env_or_cfg, "cfg", raw_env_or_cfg)
    cfg.use_privileged_base_lin_vel_obs = mode == "privileged"
    cfg.use_foot_kinematic_base_lin_vel_obs = mode == "kinematic"


def read_policy_observation(raw_env, mode: str, noise_enabled: bool) -> TensorDict:
    """Read an observation without permanently changing the task config."""
    old_privileged = raw_env.cfg.use_privileged_base_lin_vel_obs
    old_kinematic = raw_env.cfg.use_foot_kinematic_base_lin_vel_obs
    old_noise = raw_env.cfg.observation_noise_enabled
    try:
        _set_base_velocity_mode(raw_env, mode)
        raw_env.cfg.observation_noise_enabled = noise_enabled
        obs_dict = raw_env._get_observations()
    finally:
        raw_env.cfg.use_privileged_base_lin_vel_obs = old_privileged
        raw_env.cfg.use_foot_kinematic_base_lin_vel_obs = old_kinematic
        raw_env.cfg.observation_noise_enabled = old_noise
    return TensorDict(obs_dict, batch_size=[raw_env.num_envs])


def configure_commands(raw_env, forward_vel: torch.Tensor | float) -> None:
    raw_env._commands[:, 0] = 0.0
    if isinstance(forward_vel, torch.Tensor):
        raw_env._commands[:, 1] = forward_vel.to(raw_env.device)
    else:
        raw_env._commands[:, 1] = float(forward_vel)
    raw_env._commands[:, 2] = 0.0


def sample_forward_commands(raw_env, base_forward_vel: float, jitter: float) -> torch.Tensor:
    if jitter <= 0.0:
        return torch.full((raw_env.num_envs,), base_forward_vel, device=raw_env.device)
    offset = (2.0 * torch.rand(raw_env.num_envs, device=raw_env.device) - 1.0) * jitter
    return torch.full_like(offset, base_forward_vel) + offset


def rollout_verify(
    env,
    raw_env,
    policy,
    mode: str,
    forward_vel: float,
    steps: int,
    noise_enabled: bool,
) -> dict[str, float]:
    obs = read_policy_observation(raw_env, mode, noise_enabled)
    vel_y = []
    lateral_abs = []
    yaw_abs = []
    tilt = []
    base_height = []
    contact_count = []
    action_abs = []
    action_p95_abs = []
    action_max_abs = []
    for _ in range(steps):
        with torch.no_grad():
            configure_commands(raw_env, forward_vel)
            actions = policy.act_inference(obs.to(raw_env.device))
            _, _, dones, _ = env.step(actions)
            obs = read_policy_observation(raw_env, mode, noise_enabled)
            if bool(torch.any(dones).item()):
                raw_env.episode_length_buf[dones.to(raw_env.device) > 0] = 0
            vel_y.append(raw_env._robot.data.root_lin_vel_b[:, 1].detach().mean())
            lateral_abs.append(torch.abs(raw_env._robot.data.root_lin_vel_b[:, 0]).detach().mean())
            yaw_abs.append(torch.abs(raw_env._robot.data.root_ang_vel_b[:, 2]).detach().mean())
            tilt.append(torch.norm(raw_env._robot.data.projected_gravity_b[:, :2], dim=1).detach().mean())
            base_height.append(raw_env._robot.data.root_pos_w[:, 2].detach().mean())
            contact_count.append(raw_env._get_foot_contacts().float().sum(dim=1).detach().mean())
            action_abs_t = torch.abs(actions).detach()
            action_abs.append(action_abs_t.mean())
            action_p95_abs.append(action_abs_t.quantile(0.95))
            action_max_abs.append(action_abs_t.max())

    def mean(values: list[torch.Tensor]) -> float:
        return torch.stack(values).mean().item()

    return {
        "mean_vel_y": mean(vel_y),
        "mean_lateral_abs": mean(lateral_abs),
        "mean_yaw_abs": mean(yaw_abs),
        "mean_tilt": mean(tilt),
        "mean_base_height": mean(base_height),
        "mean_contact_count": mean(contact_count),
        "mean_action_abs": mean(action_abs),
        "mean_action_p95_abs": mean(action_p95_abs),
        "mean_action_max_abs": mean(action_max_abs),
    }


def main() -> None:
    print("[DISTILL] starting deployment-observation distillation", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    student_noise_enabled = not args.disable_student_observation_noise
    student_noise_samples = max(1, args.student_noise_samples)
    target_action_clip = args.target_action_clip if args.target_action_clip and args.target_action_clip > 0.0 else None

    print(f"[DISTILL] parsing task config: {args.task}", flush=True)
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.lin_vel_x_range = (0.0, 0.0)
    env_cfg.lin_vel_y_range = (args.forward_vel, args.forward_vel)
    env_cfg.ang_vel_z_range = (0.0, 0.0)
    env_cfg.rel_standing_envs = 0.0
    env_cfg.observation_noise_enabled = False
    _set_base_velocity_mode(env_cfg, args.student_base_lin_vel_observation)
    if hasattr(env_cfg, "events"):
        env_cfg.events.physics_material = None

    print("[DISTILL] loading RSL-RL config", flush=True)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device

    print("[DISTILL] creating environment", flush=True)
    raw_gym_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_gym_env, clip_actions=agent_cfg.clip_actions)
    raw_env = raw_gym_env.unwrapped
    configure_commands(raw_env, args.forward_vel)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[DISTILL] loading checkpoint: {args.checkpoint}", flush=True)
    runner.load(str(args.checkpoint), load_optimizer=False, map_location="cpu")
    student = runner.alg.policy
    teacher = copy.deepcopy(student).to(args.device)
    teacher.eval()
    student.eval()

    initial_obs = read_policy_observation(raw_env, args.student_base_lin_vel_observation, student_noise_enabled).to(
        args.device
    )
    with torch.no_grad():
        actor_obs_dim = student.actor_obs_normalizer(student.get_actor_obs(initial_obs)).shape[-1]
        action_dim = student.act_inference(initial_obs).shape[-1]

    sample_count = args.steps * args.num_envs * student_noise_samples
    dataset_obs = torch.empty((sample_count, actor_obs_dim), dtype=torch.float32, device=args.device)
    dataset_actions = torch.empty((sample_count, action_dim), dtype=torch.float32, device=args.device)
    print(
        "[DISTILL] collecting"
        f" states={args.steps * args.num_envs}"
        f" samples={sample_count}"
        f" student_obs={args.student_base_lin_vel_observation}"
        f" noise={student_noise_enabled}",
        flush=True,
    )

    write_index = 0
    action_mse_before = []
    student_obs = initial_obs
    for step in range(args.steps):
        with torch.no_grad():
            command = sample_forward_commands(raw_env, args.forward_vel, args.command_jitter)
            configure_commands(raw_env, command)
            teacher_obs = read_policy_observation(raw_env, "privileged", noise_enabled=False).to(args.device)
            teacher_actions = teacher.act_inference(teacher_obs)
            target_actions = (
                torch.clamp(teacher_actions, -target_action_clip, target_action_clip)
                if target_action_clip is not None
                else teacher_actions
            )

            clean_student_obs = read_policy_observation(
                raw_env,
                args.student_base_lin_vel_observation,
                noise_enabled=False,
            ).to(args.device)
            student_actions_before = student.act_inference(clean_student_obs)
            action_mse_before.append((student_actions_before - target_actions).square().mean().detach())

            for _ in range(student_noise_samples):
                student_obs = read_policy_observation(
                    raw_env,
                    args.student_base_lin_vel_observation,
                    noise_enabled=student_noise_enabled,
                ).to(args.device)
                actor_obs = student.actor_obs_normalizer(student.get_actor_obs(student_obs))
                end_index = write_index + args.num_envs
                dataset_obs[write_index:end_index].copy_(actor_obs)
                dataset_actions[write_index:end_index].copy_(target_actions)
                write_index = end_index

            rollout_obs = student_obs if args.rollout_action_source == "student" else teacher_obs
            rollout_policy = student if args.rollout_action_source == "student" else teacher
            rollout_actions = rollout_policy.act_inference(rollout_obs)
            if args.rollout_action_source == "teacher" and target_action_clip is not None:
                rollout_actions = torch.clamp(rollout_actions, -target_action_clip, target_action_clip)
            _, _, dones, _ = env.step(rollout_actions)
            if bool(torch.any(dones).item()):
                raw_env.episode_length_buf[dones.to(raw_env.device) > 0] = 0

        if (step + 1) % args.log_interval == 0 or step == 0:
            mean_vel_y = raw_env._robot.data.root_lin_vel_b[:, 1].mean().item()
            mean_yaw = torch.abs(raw_env._robot.data.root_ang_vel_b[:, 2]).mean().item()
            recent_mse = torch.stack(action_mse_before[-min(len(action_mse_before), args.log_interval) :]).mean().item()
            print(
                "[DISTILL][collect]"
                f" step={step + 1}/{args.steps}"
                f" vel_y={mean_vel_y:.3f}"
                f" yaw_abs={mean_yaw:.3f}"
                f" prefit_action_mse={recent_mse:.6f}",
                flush=True,
            )

    optimizer = torch.optim.Adam(student.actor.parameters(), lr=args.lr)
    student.train()
    print(
        f"[DISTILL] fitting actor epochs={args.epochs} minibatch={args.minibatch_size}",
        flush=True,
    )
    for epoch in range(args.epochs):
        permutation = torch.randperm(sample_count, device=args.device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, sample_count, args.minibatch_size):
            ids = permutation[start : start + args.minibatch_size]
            predicted_actions = student.actor(dataset_obs[ids])
            loss = (predicted_actions - dataset_actions[ids]).square().mean()
            if target_action_clip is not None and args.action_limit_loss_weight > 0.0:
                action_excess = torch.relu(torch.abs(predicted_actions) - target_action_clip)
                loss = loss + args.action_limit_loss_weight * action_excess.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.detach().item()
            batches += 1
        print(
            f"[DISTILL][fit] epoch={epoch + 1}/{args.epochs} loss={epoch_loss / max(1, batches):.7f}",
            flush=True,
        )

    student.eval()
    with torch.no_grad():
        post_pred = student.actor(dataset_obs)
        action_mse_after = (post_pred - dataset_actions).square().mean().item()
        action_mse_before_mean = torch.stack(action_mse_before).mean().item()

    env.reset()
    configure_commands(raw_env, args.forward_vel)
    clean_metrics = rollout_verify(
        env,
        raw_env,
        student,
        args.student_base_lin_vel_observation,
        args.forward_vel,
        args.verify_steps,
        noise_enabled=False,
    )
    env.reset()
    configure_commands(raw_env, args.forward_vel)
    noisy_metrics = rollout_verify(
        env,
        raw_env,
        student,
        args.student_base_lin_vel_observation,
        args.forward_vel,
        args.verify_steps,
        noise_enabled=student_noise_enabled,
    )
    print(f"[DISTILL][verify_clean] {json.dumps(clean_metrics, sort_keys=True)}", flush=True)
    print(f"[DISTILL][verify_noisy] {json.dumps(noisy_metrics, sort_keys=True)}", flush=True)

    source = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    saved_infos = {
        "source_checkpoint": str(args.checkpoint),
        "method": "privileged_teacher_to_deploy_observation_actor_distillation",
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "lr": args.lr,
        "forward_vel": args.forward_vel,
        "command_jitter": args.command_jitter,
        "target_action_clip": target_action_clip,
        "action_limit_loss_weight": args.action_limit_loss_weight,
        "student_base_lin_vel_observation": args.student_base_lin_vel_observation,
        "student_observation_noise_enabled": student_noise_enabled,
        "student_noise_samples": student_noise_samples,
        "rollout_action_source": args.rollout_action_source,
        "action_mse_before": action_mse_before_mean,
        "action_mse_after": action_mse_after,
        "verify_clean": clean_metrics,
        "verify_noisy": noisy_metrics,
    }
    saved = {
        "model_state_dict": {key: value.detach().cpu() for key, value in student.state_dict().items()},
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "iter": int(source.get("iter", 0)),
        "infos": {**(source.get("infos") or {}), "deploy_distillation": saved_infos},
    }
    torch.save(saved, args.output)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(saved_infos, indent=2, sort_keys=True) + "\n")
    env.close()
    print(f"CHECKPOINT={args.output}", flush=True)
    print(f"SUMMARY={summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
