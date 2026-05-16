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
- `trained_models/`: selected checkpoints, short videos, and strict evaluator
  outputs.
- `HANDOFF_QUAD.md`: detailed continuation notes for future Codex sessions.

## Best Checkpoints

- Main forward policy: `trained_models/forward_strict_best/model_1399.pt`
- Visual high-step forward benchmark:
  `trained_models/forward_visual_high_step/model_1558.pt`
- Best reverse smoke-test policy:
  `trained_models/reverse_straighter_best/model_1779.pt`

Install the RL package:

```bash
/workspace/isaaclab/isaaclab.sh -p -m pip install -e /workspace/insectoid_mini_quad/rl
```

See `trained_models/README.md` and `rl/PROGRESS_LOG.md` for metrics and run
history.

