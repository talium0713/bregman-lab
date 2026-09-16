"""run_canon_final.py — C3/A10 final deliverables only (does NOT rerun the §4.3 sweep):
  1) fig_permissibility_bias  — off-policy single-sample |bias| vs drift (canonical forms); RKL≡0.
  2) 2×7 standard(Amari,f'=0) vs canonical(f'=f'') off-policy π* recovery at a high-drift peak.

Run:  TWOX7_NMDP=100 python run_canon_final.py        # peak 0.9, 100 MDP draws (paper deliverable)
      PEAK=0.95 TWOX7_NMDP=100 python run_canon_final.py
"""
import json
import os

import numpy as np

import run_part3 as rp
import fig_permissibility_bias as fbias


def main():
    n = int(os.environ.get("TWOX7_NMDP", 100))
    peak = float(os.environ.get("PEAK", 0.9))
    fbias.render()                                            # deterministic, no training

    rng = np.random.default_rng(20260826)
    a0, r0, pols, gp = rp.run_2x7(peak, rng, n)
    sfx = f"_p{int(round(peak * 10))}"
    fpath = rp.fig_policy_3x7(r0, a0, pols, gp, peak, sfx)

    os.makedirs("data/tabular", exist_ok=True)
    # NOTE: mean_std returns (mean, STD) — not a CI. Store both the summary and the per-MDP arrays
    # so paired intervals can be computed without re-running.
    out = {"peak": peak, "n_mdp": n, "fig": fpath,
           "summary_is": "(mean, std) across MDPs — divide by sqrt(n) for a standard error",
           "gap": {"exact": {k: list(gp["exact"][k]) for k in rp.REGKEYS},
                   "std": {k: list(gp["std"][k]) for k in rp.REGKEYS},
                   "canon": {k: list(gp["canon"][k]) for k in gp["canon"]}},
           "per_mdp": gp["_per_mdp"],
           # MDP-0 policies + the reward draw, so fig_policy_3x7 can be re-rendered for a style
           # change without re-training (the panel only ever plots MDP 0).
           "panel_cache": {"alphas": {k: float(v) for k, v in a0.items()},
                           "rewards": np.asarray(r0).tolist(),
                           "pols": {arm: {k: np.asarray(v).tolist() for k, v in d.items()}
                                    for arm, d in zip(("exact", "std", "canon"), pols)}}}
    json.dump(out, open(f"data/tabular/canon_2x7{sfx}.json", "w"), indent=1)

    print("\npaired std - canon (same seed & data per MDP), mean [95% CI]:")
    for k in gp["canon"]:
        a = np.asarray(gp["_per_mdp"]["std"][k]); b = np.asarray(gp["_per_mdp"]["canon"][k])
        d = a - b
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {rp.SHORT[k]:6s} {d.mean():+.3f} [{d.mean()-ci:+.3f}, {d.mean()+ci:+.3f}]"
              + ("   (excludes 0)" if abs(d.mean()) > ci else "   (includes 0)"))

    print("\nstd   :", {rp.SHORT[k]: round(gp['std'][k][0], 3) for k in rp.REGKEYS})
    print("canon :", {rp.SHORT[k]: round(gp['canon'][k][0], 3) for k in gp['canon']})
    order = sorted(gp['canon'].items(), key=lambda kv: kv[1][0])
    print("canon ranked:", " < ".join(f"{rp.SHORT[k]}{v[0]:.3f}" for k, v in order))
    print("saved", fpath, "+ data/tabular/canon_2x7" + sfx + ".json")


if __name__ == "__main__":
    main()
