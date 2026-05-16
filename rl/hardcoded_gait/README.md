# Hard-Coded Foot-Line Quadruped Walk Demo

This folder is separate from the direct PPO/RL setup. It loads the same current
insectoid mini quad asset and drives the four walking legs with an analytic
open-loop static walk in foot space.

The controller uses the current standing pose as the neutral pose, then cycles
the legs one at a time:

- `BR`
- `ML`
- `BL`
- `MR`

For each leg, stance holds the foot target fixed in world coordinates. During
swing, the foot target moves on a straight line parallel to world Y and lifts in
Z. The script uses the simulator Jacobian to convert each desired foot target
into joint targets.

Current rendered demo:

- video: `outputs/foot_line_walk_10cm_y_inverted_v2/hardcoded_foot_line_walk.mp4`
- initial snapshot: `outputs/foot_line_walk_10cm_y_inverted_v2/hardcoded_foot_line_walk_initial.png`
- final snapshot: `outputs/foot_line_walk_10cm_y_inverted_v2/hardcoded_foot_line_walk_final.png`

Current foot trajectory:

- one leg swings at a time: `BR -> ML -> BL -> MR`
- step line length: `10 cm`
- swing height: `3 cm`
- the first positive-Y attempt moved the base backward, so this render uses the
  inverted Y direction (`--step-length -0.10`) and produced positive base-Y
  displacement in simulation.

Command used:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p /workspace/insectoid_mini_quad_rl/hardcoded_gait/render_hardcoded_quad_gait.py \
  --duration 10.0 \
  --fps 25 \
  --cycle-time 2.8 \
  --swing-fraction 0.22 \
  --step-length -0.10 \
  --step-height 0.03 \
  --ik-gain 0.7 \
  --ik-damping 0.04 \
  --max-joint-step-deg 4.0 \
  --eye 3.0 -2.7 1.45 \
  --lookat 0.0 0.0 0.13 \
  --output-dir /workspace/insectoid_mini_quad_rl/hardcoded_gait/outputs/foot_line_walk_10cm_y_inverted_v2
```
