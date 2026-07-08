# Repository Layout

This repo separates source, training artifacts, extracted motion data, and
generated videos so future Isaac Lab work does not become mixed with render
experiments.

## Source And Robot Assets

- `URDF_description/`: quad robot URDF, USD, and local mesh files.
- `scripts/`: asset generation and verification scripts.
- `rl/`: Isaac Lab task package, RSL-RL config, training/playback scripts, and
  development progress log.

## Training Artifacts

- `trained_models/`: selected checkpoints, exported policies, and evaluator
  text outputs.
- `outputs/training_rollouts/`: canonical videos rendered from trained
  checkpoints.

Do not put new videos under `trained_models/`. Checkpoints and exports belong
there; generated visual artifacts belong under `outputs/`.

## Motion Data

- `data/extracted_gaits/`: policy rollout captures and phase-averaged gait data.
- `data/motion_primitives/`: curated hardcoded or post-processed primitives.
- `data/trajectories/`: compact trajectory captures used by render scripts.
- `data/analysis/`: CSV/JSON diagnostics such as attitude and telemetry
  analysis.

Scripts may depend on files in `data/`, so this folder should be treated as
project data rather than disposable output.

## Render Outputs

- `outputs/renders/`: generated Isaac Sim presentation videos, contact sheets,
  logs, smoke-test summaries, and tuning sweeps.
- `outputs/training_rollouts/`: selected rollout videos tied to trained
  checkpoints.

Most generated outputs are ignored by Git. Keep final videos on disk here, and
only force-add a render artifact when there is a deliberate reason to version
it.

## Documentation And Handoff Material

- `docs/handoff/`: notes for continuing training and model handoff.
- `packages/handoff/`: generated zip archives for sharing with collaborators.
- `external/references/`: local third-party/reference files downloaded for
  analysis. These can be large and should not be treated as core source.

## Current Key Files

- Best forward checkpoint:
  `trained_models/forward_strict_best/model_1399.pt`
- Canonical forward rollout:
  `outputs/training_rollouts/forward_strict_best/forward_strict_best_model_1399.mp4`
- Current deploy-adapted quad checkpoints:
  `trained_models/deploy_quad_kinematic/`
- Material relay demo:
  `rl/scripts/render_material_relay_handoff.py`
- Latest material relay render output:
  `outputs/renders/material_relay_handoff_v10_short_handoff_stages.mp4`
