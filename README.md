# Insectoid Mini Quad

Four-walking-leg Isaac Lab variant of the insectoid mini robot.

The full front legs are kept as fixed swept-forward arms. Locomotion uses only
the middle and rear four legs, with fixed spherical foot links at the ends of
the four walking tibias.

## Layout

- `URDF_description/`: generated URDF/USD asset and mesh files.
- `scripts/`: asset generation and verification scripts.
- `rl/`: Isaac Lab DirectRLEnv package, RSL-RL config, diagnostics, playback,
  and progress log.
- `trained_models/`: selected policy checkpoints, exported policies, and strict
  evaluator outputs.
- `data/`: extracted trajectories, motion primitives, and analysis data used by
  scripted render demos.
- `outputs/`: generated videos, images, logs, and summaries. Canonical rollout
  videos live under `outputs/training_rollouts/`.
- `docs/`: repo layout notes and handoff documentation for future sessions.
- `packages/`: generated handoff archives.
- `external/`: local third-party/reference material that is useful during
  development but is not core source.

See `docs/REPO_LAYOUT.md` for the detailed organization rules.

## Best Checkpoints

- **Selected favorite forward locomotion policy**:
  `trained_models/forward_strict_best/model_1399.pt`
- Visual high-step forward benchmark:
  `trained_models/forward_visual_high_step/model_1558.pt`
- Best reverse smoke-test policy:
  `trained_models/reverse_straighter_best/model_1779.pt`

Use `model_1399.pt` as the final mini-quad forward locomotion baseline. It is
the highest-scoring and most reliable policy from this project.

Install the RL package:

```bash
/workspace/isaaclab/isaaclab.sh -p -m pip install -e /workspace/insectoid_mini_quad/rl
```

See `trained_models/README.md`, `docs/handoff/HANDOFF_QUAD.md`, and
`rl/PROGRESS_LOG.md` for metrics and run history.
