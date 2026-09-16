"""rerender_3x7.py — re-render fig_policy_3x7 from the cached MDP-0 panel data in
data/tabular/canon_2x7{sfx}.json WITHOUT re-training. run_canon_final.py stores `panel_cache`
(MDP-0 policies for all three arms + the reward draw + calibrated alphas) exactly so a style or
labelling change to the 3x7 panel can be re-rendered in seconds instead of re-running 100 MDPs.

Run from python/:   python rerender_3x7.py            # default _p9 (the f13 deliverable)
                    python rerender_3x7.py _p9 _p7
"""
import json
import sys

import numpy as np

import run_part3 as rp

ARMS = ("exact", "std", "canon")


def rerender(sfx):
    d = json.load(open(f"data/tabular/canon_2x7{sfx}.json"))
    pc = d["panel_cache"]
    alphas = {k: float(v) for k, v in pc["alphas"].items()}
    rewards = np.asarray(pc["rewards"])
    pols = tuple({k: np.asarray(v) for k, v in pc["pols"][arm].items()} for arm in ARMS)
    gaps = {arm: {k: tuple(v) for k, v in d["gap"][arm].items()} for arm in ARMS}
    out = rp.fig_policy_3x7(rewards, alphas, pols, gaps, d["peak"], sfx)
    print(f"[rerendered] {out}  ·  peak {d['peak']}, {d['n_mdp']} MDPs, from cache (no training)")
    return out


if __name__ == "__main__":
    for a in (sys.argv[1:] or ["_p9"]):
        rerender(a if a.startswith("_") else f"_{a}")
