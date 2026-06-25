# Data

Project data used by analysis and scripted demos.

- `extracted_gaits/`: captured policy rollouts and phase-averaged joint data.
- `motion_primitives/`: curated forward, reverse, and sideways primitives.
- `trajectories/`: compact trajectory tapes consumed by render scripts.
- `analysis/`: attitude, telemetry, and diagnostic CSV/JSON outputs.

Unlike `outputs/`, files here may be script inputs and should be preserved when
they define a reusable primitive or analysis baseline.
