# Killarney — tabular reproduction

Self-contained scaffold for re-running the **tabular** experiments on the Vector/Killarney
cluster (Alliance). This folder keeps the cluster work separate from local dev: job scripts,
logs, and a collected results snapshot all live here. The actual experiment code stays in
`../python/`; these scripts just drive it.

**CPU-only.** The tabular code is pure NumPy (no torch, no autograd). GPU adds nothing and would
only lower scheduling priority — so the array job requests **no `--gres`**. A separate optional
`jobs/smoke_gpu.slrm` exists only to test the GPU path before the LLM Stage A work.

## Layout

```
killarney/
  setup_env.sh            one-time CPU venv (numpy + matplotlib) at ../venv, login node
  experiments.list        data-generation commands, one per array task (checked items)
  run_local.sh            no-SLURM dry run to verify the pipeline locally
  jobs/
    01_data_array.slrm    phase 1: CPU array job, runs each experiments.list line
    02_figs.slrm          phase 2: regenerate figures + collect snapshot to results/
    smoke_gpu.slrm        optional GPU env test (unrelated to tabular)
  logs/                   sbatch logs (%x-%A_%a.out / %x-%j.out)
  results/                collected figures + run manifests per run_<timestamp>/
```

Raw data lands in `../python/data/tabular/run_*/` (regenerable); `results/` holds the collected
figures + manifests (git commit, seeds, hyperparams) as the distinguishable cluster snapshot.

## Run it

```bash
# 0) on Killarney, clone/pull the repo under your account, then:
cd ~/projects/aip-rudner/$USER/bregman-lab/killarney

# 1) one-time env (login node)
bash setup_env.sh

# 2) phase 1 — data (13 array tasks). Recompute the array size if you edit experiments.list:
grep -cvE '^[[:space:]]*(#|$)' experiments.list        # -> keep --array=1-N in 01_data_array.slrm in sync
sbatch jobs/01_data_array.slrm                         # prints: Submitted batch job <ARRAY_ID>

# 3) phase 2 — figures + snapshot, chained after phase 1
sbatch --dependency=afterok:<ARRAY_ID> jobs/02_figs.slrm

# monitor
squeue --me
tail -f logs/tab_data-<ARRAY_ID>_1.out
seff <jobid>                                           # after: check actual mem/cpu use
```

Or verify the whole thing locally first (no cluster): `bash run_local.sh` (or `--figs-only`).

## What's included (from the Notion checklist, 2026-07-20)

Checked → included here. **A2 (ε ablation) was left unchecked, so it is excluded.**

| id | experiment | driver | figures |
|----|------------|--------|---------|
| M1 | single-state inner-term variance (std vs n ∝ 1/√n) | `fig_mechanism.py` (phase 2, no data run) | `mechanism_single_state.png` |
| M2 | trajectory variance (std vs H ∝ √(H−1)) | `fig_mechanism.py` | `mechanism_trajectory.png` |
| H1 | headline: MC budget × 3 regimes, peak 0.7, 100 MDP | `run_tabular.py` → `fig_tabular.py` | `tabular_headline_p70.png`, `tabular_offpolicy.png`, `tabular_calibration.{csv,tex}` |
| A1 | peak/sharpness ablation {0.6, 0.8} | `run_tabular.py` → `fig_tabular.py` | `tabular_offpolicy_peaks.png`, `tabular_headline_p60/p80.png` |
| A3 | horizon H ablation {2,3,4,5,6} | `run_tabular.py --depth` → `fig_horizon.py` | `horizon_nmc1.png`, `horizon_nmc16.png` |
| A4 | α-div sweep, standard norm (baseline), peaks {0.6,0.7,0.8} | `run_adiv.py` → `fig_adiv.py` | `adiv_curve/morph_p{60,70,80}.png` |
| A4b | α-div KL-consistent norm (kln, diagnostic) | `run_adiv.py --kl-norm` → `fig_adiv_compare.py`, `fig_adiv_invariant.py` | `adiv_*_kln.png`, `adiv_compare_p80.png`, `adiv_invariant.png` |
| E-CI | CI sensitivity n=10 vs n=100 | **no generator script in repo** | `tabular_ci_n10_vs_n100.png` |

**CI figure caveat:** `tabular_ci_n10_vs_n100.png` has no reproducible generator in `../python/`
(it was made ad-hoc). Phase 2 skips it. If you want it reproducible, it's an easy add — a small
script that runs the headline at `--n-mdp 10` and `--n-mdp 100` and overlays the two CIs. Ask and
I'll write it.

## Notes

- Each array task runs `run_tabular.py`/`run_adiv.py` with `--jobs $SLURM_CPUS_PER_TASK` (its 100
  MDPs parallelized over the requested cores). 13 tasks × 8 cores schedule independently.
- `fig_tabular.py` **merges all** `data/tabular/run_*/results.json` keyed by calibration peak, so a
  clean data dir per fresh reproduction is safest.
- Divergence colors were reassigned to the "Matching Gradient" palette in `../python/regularizers.py`
  (`COLORS`); all figures pick it up on regeneration.
- Legacy `run_part3.py` (`part3_*.png`) is **not** part of this reproduction — order-dependent single
  RNG, superseded by the `seeds.py` + `run_tabular.py` pipeline.
