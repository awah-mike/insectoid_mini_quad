# model_1399_forward_canonical_sideways_bidirectional_s25_v1

Bidirectional sideways translation primitives derived from the cleaned canonical forward trajectory.

Source:

- Forward source label: `model_1399_forward_recycled_stride_v1`
- Base checkpoint: `trained_models/forward_strict_best/model_1399.pt`
- Captured cleaned trajectory: `data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz`
- Replay source: realized joint positions, `source=actual`
- Replay window: `0.72s..1.30s`
- Playback: forward order, cycled, `2.5x`

Shared method:

- Convert the canonical forward foot Y excursion into body-X foot excursion.
- Hold body-frame Y velocity at `0.0 m/s` with the root velocity servo.
- Use a body-frame X velocity servo for lateral translation.
- Keep the small scheduled lift from recorded foot contacts.
- Use coxa yaw feedback and yaw-rate damping; direct root yaw pose servo was tested and rejected because it increased path curvature.

Selected outputs:

- Cleaned negative-X: `outputs/renders/model_1399_forward_canonical_sideways_xneg_clean_s25_front_iso.mp4`
- Positive-X opposite direction: `outputs/renders/model_1399_forward_canonical_sideways_xpos_signpos_s25_front_iso.mp4`

Measured metrics over 6 seconds:

| Primitive | Body X vel | World X/Y disp | Roll/Pitch/Yaw p2p | Non-foot contact |
| --- | ---: | ---: | ---: | ---: |
| xneg clean | `-0.300 m/s` | `-1.560 / 0.343 m` | `4.60 / 4.13 / 24.31 deg` | `0.0000` |
| xpos | `+0.350 m/s` | `+1.819 / 0.267 m` | `7.22 / 4.53 / 22.13 deg` | `0.0000` |

Notes:

- The old fast negative-X render is still available at `outputs/renders/model_1399_forward_canonical_sideways_xneg_s25_front_iso.mp4`.
- The cleaner negative-X version reduces roll and slows the primitive slightly.
- The positive-X direction needed the standard yaw feedback sign in full render mode; a flipped-sign version looked better in no-video sweeps but did not win in the final render.
- These remain post-model visualization primitives, not trained lateral locomotion policies.
