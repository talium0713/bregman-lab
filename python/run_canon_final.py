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
    a0, r0, sp, cp, gp = rp.run_2x7(peak, rng, n)
    sfx = f"_p{int(round(peak * 10))}"
    fpath = rp.fig_policy_2x7(r0, a0, sp, cp, gp, peak, sfx)

    os.makedirs("data/tabular", exist_ok=True)
    out = {"peak": peak, "n_mdp": n, "fig": fpath,
           "gap": {"std": {k: list(gp["std"][k]) for k in rp.REGKEYS},
                   "canon": {k: list(gp["canon"][k]) for k in gp["canon"]}}}
    json.dump(out, open(f"data/tabular/canon_2x7{sfx}.json", "w"), indent=1)

    print("\nstd   :", {rp.SHORT[k]: round(gp['std'][k][0], 3) for k in rp.REGKEYS})
    print("canon :", {rp.SHORT[k]: round(gp['canon'][k][0], 3) for k in gp['canon']})
    order = sorted(gp['canon'].items(), key=lambda kv: kv[1][0])
    print("canon ranked:", " < ".join(f"{rp.SHORT[k]}{v[0]:.3f}" for k, v in order))
    print("saved", fpath, "+ data/tabular/canon_2x7" + sfx + ".json")


if __name__ == "__main__":
    main()
