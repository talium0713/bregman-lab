# Prompt: re-run the paper's legacy tree sweeps on the layered MDP

Paste this to an agent working in `bregman-lab/python`. It comes from the paper repo
(`Off_policy_admissability/main.tex`) after round 0916-B, where option (c) was chosen: the toy
tree MDP leaves the paper and every result that still depends on it is regenerated on the layered
environment.

## Why

The paper's Appendix C now describes one environment only, the layered ε-stochastic MDP that
`mdp.py` implements. Seven figures, two numeric tables, and two hyperparameter rows still come from
the older deterministic complete-tree code, so the text and the plots currently describe different
environments. The docstring in `mdp.py` already flags the mismatch.

## Target configuration, already verified against `data/tabular/run_*/results.json`

```
env    = {depth: 4 decision layers, SN: 3 states per layer, NA: 3 actions,
          eps: 0.2, transitions depend on the action only,
          reward: Uniform(-0.8, 0.8) on every (layer, state, action), gamma: 0.9}
start  = uniform over the three layer-0 states
pi_ref = uniform at every state
oracle = Bradley-Terry on gamma-discounted trajectory return differences, temperature 1
runs   = 100 independently drawn MDPs, one training seed each
weight = regularization weight calibrated per (regularizer, target peak) by bisection,
         not a fixed beta
adiv   = alpha = 0.5   (DEFAULT_ADIV_A)
roster = RKL, alpha-div, FKL, JS, squared Hellinger, Pearson chi^2, squared Euclidean
```

Do not change any of these to make a plot look better. If a run needs a different value, say so
instead of silently diverging, because the paper quotes this configuration verbatim.

## What has to be regenerated

Each row is a figure or table the paper still includes. "Legacy source" is the tree-based artifact
it uses today. "Likely layered source" is an existing script in this repo that already targets the
layered environment, so check it before writing anything new.

| Paper location | What it must show | Legacy source | Likely layered source |
| --- | --- | --- | --- |
| D, single-state variance | std of the n-sample inner-term estimator against n, per regularizer, KL exactly zero | `exp1_variance_vs_n.pdf` | `fig_mechanism.py` → `mechanism_single_state` |
| D, trajectory compounding | per-trajectory std of the summed inner term against horizon, sqrt(H-1) growth | `exp2_trajectory_mc.pdf` | `fig_mechanism.py` → `mechanism_trajectory`, or `fig_horizon.py` |
| D, recovery vs budget | policy gap against n_mc, on-policy and off-policy panels | `exp23_policy_gap_tv_calibrated_withstd.pdf` | `run_part3.py` / `fig_tabular.py` |
| D, per-state anatomy | see the open question below | `exp25_leaf_anatomy_calibrated_on.pdf` | none yet |
| D, non-uniform reference | the same recovery curves under a peaked reference | `exp24_policy_gap_tv_calibrated_peaked_withstd.pdf` | `run_part3.py` with a peaked `pi_ref` |
| D, alpha-family sweep | policy gap along the alpha family against n_mc | `exp27_alpha_div_sweep.pdf` | `fig_adiv.py` / `run_adiv.py` |
| C, calibration sweep | policy peak against the regularization weight, with the calibrated anchors | `exp22_alpha_peak_match_nostd.pdf` | `experiments.py` → `calibrate` |
| C, calibration table | calibrated weight per regularizer, median and range | table in the paper | same as above |
| D, two numeric tables | the numbers behind the first two rows at three budgets and four horizons | tables in the paper | same as the first two rows |
| C, hyperparameter table | replace the rows `W = 3` "tree width" and `H = 3` "tree depth" with the layered parameters | table in the paper | `mdp.py` defaults |

## One open design question, answer it before running

The per-leaf anatomy plots 27 leaves in BFS order, which the layered MDP does not have. Two
candidates, pick one and say which:

1. per (layer, state) cells, plotting the recovered policy against the target at each of the twelve
   cells, which is the direct analogue of "what the policy gap integrates over";
2. per terminal state, plotting the reach probability of each layer-3 state, which keeps the
   reach-probability reading of the original but has only three columns.

Option 1 looks closer to what the paper claims the figure shows.

## Output contract

- Export through `export_for_paper.py` into `figure4overleaf/` with a JSON sidecar that records the
  git commit, the script, and the full config dict above.
- Name the files so the paper can tell them apart from the legacy set, for example
  `t01_single_state_variance`, `t02_trajectory_compounding`, `t03_recovery_vs_nmc`,
  `t04_state_anatomy`, `t05_peaked_reference`, `t06_alpha_sweep`, `t07_calibration`.
- PDF and PNG for each, and print the numbers that the paper quotes in prose, so the text can be
  updated from the run rather than from the old values.
- Every panel that reports a policy gap needs the gap printed on it and a confidence interval over
  the 100 MDPs, matching how `f02_tabular_recovery` and `f13_policy_3x7_canonical` already do it.

## Acceptance check

Re-run `python -c "import json,glob; print(json.load(open(glob.glob('data/tabular/run_*/results.json')[0]))['env'])"`
and confirm the printed env matches the configuration block above before exporting anything.
