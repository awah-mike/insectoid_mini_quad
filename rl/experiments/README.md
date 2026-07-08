# Experiment Harness

Automates the train -> evaluate -> ledger loop that was previously manual.
The driver is `rl/scripts/run_experiment_batch.py`; batches are JSON specs in
this folder; every scored run is appended to `EXPERIMENTS.tsv`.

## Usage

```bash
cd /home/ubuntu/insectoid_mini_quad
python3 rl/scripts/run_experiment_batch.py --spec rl/experiments/<batch>.json
```

The driver shells out to `isaaclab.sh` for each stage, so it can run as a
plain background process. Useful flags:

- `--dry-run` prints every command without executing.
- `--only <id> [<id> ...]` runs a subset of the experiments in the spec.

Per (experiment, seed) the harness:

1. trains with `rl/scripts/train_rsl_rl.py` (`--run_name <batch>_<id>_s<seed>`),
2. finds the run dir under `logs/rsl_rl/` and its highest `model_*.pt`,
3. scores it with the strict evaluator (`rl/scripts/evaluate_rsl_rl.py`),
   writing `eval_harness_model_<iter>.txt` into the run dir,
4. optionally renders a review video (`"render": true`),
5. appends a row to `EXPERIMENTS.tsv`.

Console logs for every stage go to `outputs/harness_logs/<run_name>.log`.

## Spec format

`defaults` apply to all experiments; per-experiment keys override them.
`overrides` are Hydra overrides passed verbatim to Isaac Lab's `train.py`
(`env.*` for `DirectRLEnvCfg` fields, `agent.*` for the PPO runner cfg).
`resume` is `{"load_run": ..., "checkpoint": ...}`; the checkpoint may be an
absolute path (the train wrapper supports it).

Caution: overrides that change the network (observation normalization,
observation size) are incompatible with `resume` from a checkpoint trained
without them — use fresh runs for those.

## Ledger

`EXPERIMENTS.tsv` columns mirror the strict evaluator's headline metrics
(`TASK_SCORE`, forward velocity, stride, touchdown rate, contact imbalance,
sub-scores) plus run dir, checkpoint, overrides, and status. The keep/discard
rule from the parent project still applies: a candidate must beat the baseline
on the evaluator *and* look right in a rendered video before promotion.

Reference baselines for comparison:

- `trained_models/forward_strict_best/model_1399.pt` — TASK_SCORE 0.9211
  (Flat-Play, 0.30 m/s, 800 steps, 64 envs).
- `outputs/harness_logs/validate_model_1399_eval.txt` — harness re-score of the
  same checkpoint on this machine.

## Batches

- `plumbing_test.json` — tiny end-to-end pipeline check (3 iterations); not a
  real experiment.
- `quick_wins_gait_refine.json` — first optimization batch on the GaitRefine
  task: observation normalization, 48-step rollout horizon (covers ~a full
  gait cycle at 50 Hz instead of 0.48 s), and the combination, each at 2 seeds
  against a fresh baseline control.
