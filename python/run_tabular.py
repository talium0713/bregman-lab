"""
run_tabular.py — the REPRODUCIBLE tabular experiment driver for the paper.

Unlike run_part3.py (one shared RNG, order-dependent, non-addressable "seeds"), every cell here
draws from an independent stream derived from a recorded ROOT_SEED via seeds.rng_for, so:
  * a single (mdp, peak, regime, Omega, n_mc, seed) cell is reproducible in isolation,
  * results are ORDER-INDEPENDENT (axes/grids can change without perturbing other cells),
  * "100 seeds" is a real average over 100 independent, named streams.

DESIGN (locked with the author):
  * variance source = a FIXED set of MDPs x many TRAINING seeds  -> isolates the inner-term /
    estimation noise from MDP-to-MDP variance (the "noise-free inner term" claim, directly).
  * three data regimes, treated equally:  off (pure off-policy, n_mc irrelevant),
    off_on (Dyna: off-policy states, fresh n_mc resample), on (on-policy lagged sampler).
  * environment = the stochastic layered MDP (eps=0.2), matching the JS/Streamlit suites.

Output is self-describing data only (no figures): out/manifest.json (ROOT_SEED + every
hyperparameter + git commit) and out/results.json (per-cell mean/std/sem/CI + seed-0 curve &
policy + pi* target).  Figures are a separate step that reads results.json.

Run:
    python run_tabular.py --quick                 # ~1 min smoke + reproducibility self-check
    python run_tabular.py                         # full headline matrix (see DEFAULTS below)
    python run_tabular.py --n-seeds 100 --jobs 8  # explicit
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np

from regularizers import REG, REGKEYS, is_admissible, COLORS, SHORT
from mdp import solve_dp, uniform_pis, new_rewards, state_occupancy, SN, NA
from dpo import TrainConfig, make_dataset, train_one
from experiments import calibrate, peakiness
from inner_term import C_exact
from seeds import ROOT_SEED, REGIME_IDX, REGKEY_IDX, rng_for

# ───────────────────────────── fixed environment / training defaults ─────────────────────────────
GAMMA, EPS, DEPTH = 0.9, 0.2, 4
BATCH, LR, NPAIRS = 16, 0.08, 6000
ON_NPAIRS = 600            # on-policy regenerates this many pairs per snapshot (kept << NPAIRS for speed)
RESAMPLE_EVERY = 25


@dataclass
class RunConfig:
    root: int = ROOT_SEED
    n_mdp: int = 3                 # fixed MDPs (each = a recorded reward draw)
    n_seeds: int = 100             # training seeds per cell
    peaks: tuple = (0.7,)          # calibration anchors
    nmc: tuple = (1, 4, 16)        # MC budget grid (off uses only n_mc=1: inner term ignores it)
    regimes: tuple = ("off", "off_on", "on", "exact")   # exact (A6): closed-form inner term, tabular only
    steps: int = 400
    gamma: float = GAMMA
    eps: float = EPS               # rollout/target transition noise (single-eps runs)
    # ε-ablation: sweep the rollout/target ε while holding α FIXED to the calibration done at
    # `calib_eps`.  eps_list None ⇒ ordinary single-eps run (calibrate at `eps`).  When set,
    # α is calibrated ONCE at calib_eps (default = eps) so only the dynamics change across the sweep.
    eps_list: tuple = None
    calib_eps: float = None
    depth: int = DEPTH
    batch: int = BATCH
    lr: float = LR
    npairs: int = NPAIRS
    on_npairs: int = ON_NPAIRS
    resample_every: int = RESAMPLE_EVERY


# ───────────────────────────── one config = one (cell-block of n_seeds trainings) ─────────────────────────────
def _nmc_grid(regime: str, nmc: tuple) -> tuple:
    """The off regime's inner term uses the single logged a' (n_mc is ignored), so sweeping it
    would only duplicate identical work -> collapse to a single point."""
    return (1,) if regime in ("off", "exact") else tuple(nmc)   # exact: closed form, n_mc irrelevant


def _train_cell(rc: RunConfig, rewards, alpha, sol, data, mi, pi, peak, regime, rk, ri, nm,
                eps_roll, ei=None) -> dict:
    """Train `n_seeds` independent seeds for ONE cell, training/evaluating at `eps_roll` (the
    rollout/target transition noise) with the supplied `alpha` (fixed across an ε sweep).  `ei` is
    the ε index for the rng key (None ⇒ single-eps run, keys unchanged for back-compat).  Heavy
    detail (curve/policy/π*) is kept only for the representative MDP mi==0 & first ε."""
    keep = (mi == 0 and ei in (None, 0))
    finals = []
    curve0 = pol0 = None
    for sd in range(rc.n_seeds):
        key = ((mi, pi, REGIME_IDX[regime], ri, nm, sd) if ei is None
               else (mi, pi, ei, REGIME_IDX[regime], ri, nm, sd))
        trng = rng_for(rc.root, "train", *key)
        cfg = TrainConfig(gamma=rc.gamma, steps=rc.steps, lr=rc.lr, batch=rc.batch, n_mc=nm,
                          policy_mode=regime, resample_every=rc.resample_every, n_pairs=rc.on_npairs)
        curve, final, pol = train_one(rk, rewards, alpha, data, cfg, eps_roll, trng)
        finals.append(float(final))
        if keep and sd == 0:
            curve0, pol0 = [float(c) for c in curve], pol.tolist()

    finals = np.asarray(finals, float)
    n = len(finals)
    mean = float(finals.mean())
    std = float(finals.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    return dict(
        mi=mi, pi=pi, peak=peak, eps=float(eps_roll), regime=regime, rk=rk, nm=nm, alpha=float(alpha),
        peakiness=float(peakiness(rk, rewards, alpha, rc.gamma, eps_roll)),
        C_exact=float(C_exact(rk, sol.pistar[0, 0])), admissible=bool(is_admissible(rk)),
        mean=mean, std=std, sem=sem, ci95=float(1.96 * sem),
        finals=finals.tolist(), curve0=curve0, pol0=pol0,
        pistar=sol.pistar.tolist() if keep else None,
    )


def run_config(task: dict, rc_d: dict) -> dict:
    """Single-cell entry point (used by the reproducibility self-check). Self-contained: rebuilds
    its own reward draw and preference data from rng_for, so it is order-independent."""
    rc = RunConfig(**rc_d)
    mi, pi, peak, regime, rk, ri, nm = (task["mi"], task["pi"], task["peak"],
                                        task["regime"], task["rk"], task["ri"], task["nm"])
    rewards = new_rewards(rc.depth, rng_for(rc.root, "reward", mi))
    alpha = calibrate(rewards, peak, rc.gamma, rc.eps)[rk]
    sol = solve_dp(rk, rewards, uniform_pis(rc.depth), alpha, rc.gamma, rc.eps)
    data = (make_dataset(rewards, rc.eps, rc.gamma, rng_for(rc.root, "data_off", mi, pi), rc.npairs)
            if regime in ("off", "off_on", "exact") else None)   # exact reads the logged states like off
    return _train_cell(rc, rewards, alpha, sol, data, mi, pi, peak, regime, rk, ri, nm, rc.eps)


def run_mdp_block(block: dict, rc_d: dict) -> list:
    """Parallel unit = one (mdp, peak): build the reward draw and calibration ONCE, then run every
    (ε, regime, Omega, n_mc) cell for it.  α is calibrated at `calib_eps` (= eps unless overridden),
    so an ε sweep holds the regularization strength fixed while the dynamics vary."""
    rc = RunConfig(**rc_d)
    mi, pi, peak = block["mi"], block["pi"], block["peak"]
    rewards = new_rewards(rc.depth, rng_for(rc.root, "reward", mi))
    sweep = rc.eps_list is not None
    eps_vals = tuple(rc.eps_list) if sweep else (rc.eps,)
    calib_eps = rc.calib_eps if rc.calib_eps is not None else rc.eps
    alphas = calibrate(rewards, peak, rc.gamma, calib_eps)        # FIXED across the ε sweep
    cells = []
    for ei, eps_roll in enumerate(eps_vals):
        sols = {rk: solve_dp(rk, rewards, uniform_pis(rc.depth), alphas[rk], rc.gamma, eps_roll)
                for rk in REGKEYS}
        for regime in rc.regimes:
            if regime in ("off", "off_on", "exact"):   # exact reads the logged states like off
                dkey = (mi, pi, ei) if sweep else (mi, pi)
                data = make_dataset(rewards, eps_roll, rc.gamma,
                                    rng_for(rc.root, "data_off", *dkey), rc.npairs)
            else:
                data = None
            for ri, rk in enumerate(REGKEYS):
                for nm in _nmc_grid(regime, rc.nmc):
                    cells.append(_train_cell(rc, rewards, alphas[rk], sols[rk], data,
                                             mi, pi, peak, regime, rk, ri, nm,
                                             eps_roll, ei if sweep else None))
    return cells


# ───────────────────────────── driver ─────────────────────────────
def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(__file__) or ".",
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run(rc: RunConfig, out_dir: str, jobs: int) -> dict:
    """Run the matrix as one parallel task per (mdp, peak) block, saving results.json incrementally
    so progress can be inspected mid-run.  manifest.json carries status running/complete + counts."""
    os.makedirs(out_dir, exist_ok=True)
    blocks = [dict(mi=mi, pi=pi, peak=peak)
              for mi in range(rc.n_mdp) for pi, peak in enumerate(rc.peaks)]
    n_eps = len(rc.eps_list) if rc.eps_list else 1
    cells_per_block = n_eps * sum(len(REGKEYS) * len(_nmc_grid(reg, rc.nmc)) for reg in rc.regimes)
    n_cells = len(blocks) * cells_per_block
    n_trains = n_cells * rc.n_seeds
    print(f"[run] {len(blocks)} MDP-blocks x {cells_per_block} cells x {rc.n_seeds} seeds "
          f"= {n_trains} trainings  (jobs={jobs})")
    print(f"      root={rc.root} mdps={rc.n_mdp} peaks={rc.peaks} regimes={rc.regimes} "
          f"nmc={rc.nmc} steps={rc.steps}")
    if rc.eps_list:
        print(f"      ε-sweep={rc.eps_list}  (α fixed via calib_eps="
              f"{rc.calib_eps if rc.calib_eps is not None else rc.eps})")

    rc_d = asdict(rc)
    manifest = dict(
        status="running", root_seed=rc.root, config=rc_d,
        n_blocks=len(blocks), blocks_done=0, n_cells=n_cells, n_trainings=n_trains,
        env=dict(gamma=rc.gamma, eps=rc.eps, depth=rc.depth, SN=SN, NA=NA, reward="Uniform(-0.8,0.8)"),
        regkeys=REGKEYS, colors=COLORS, short=SHORT, regime_idx=REGIME_IDX, regkey_idx=REGKEY_IDX,
        git_commit=_git_commit(), created_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
    )

    def save(results, status, done):
        manifest["status"], manifest["blocks_done"] = status, done
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump({"manifest": manifest, "results": results}, f)

    results, t0, done = [], time.time(), 0
    save(results, "running", 0)

    def log_block(b):
        nonlocal done
        done += 1
        el = time.time() - t0
        eta = el / done * (len(blocks) - done)
        print(f"  [{done}/{len(blocks)} blocks] mdp{b['mi']} peak{b['peak']} | "
              f"{len(results)}/{n_cells} cells | elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)

    if jobs == 1:
        for b in blocks:
            results.extend(run_mdp_block(b, rc_d))
            log_block(b)
            save(results, "running", done)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_mdp_block, b, rc_d): b for b in blocks}
            for fut in as_completed(futs):
                results.extend(fut.result())
                log_block(futs[fut])
                if done % 5 == 0 or done == len(blocks):     # snapshot every 5 blocks (+ at end)
                    save(results, "running", done)

    save(results, "complete", done)
    print(f"[saved] {out_dir}/manifest.json + results.json  ({len(results)} cells)")
    return {"manifest": manifest, "results": results}


# ───────────────────────────── reproducibility self-check ─────────────────────────────
def repro_check(rc: RunConfig) -> None:
    """Prove the three properties: identical re-run, order-independence, and seed-distinctness."""
    print("\n[repro] reproducibility self-check")
    rc_d = asdict(rc)
    t = dict(mi=0, pi=0, peak=rc.peaks[0], regime="off_on", rk="js", ri=REGKEY_IDX["js"], nm=4)
    a = run_config(t, rc_d)["finals"]
    b = run_config(t, rc_d)["finals"]
    assert a == b, "re-running the SAME cell must give identical finals"
    print(f"  PASS  identical re-run  (cell off_on/JS/nmc=4, {len(a)} seeds bit-identical)")

    # order-independence: a different cell in between must not change this cell's result
    _ = run_config(dict(mi=0, pi=0, peak=rc.peaks[0], regime="off", rk="kl",
                        ri=REGKEY_IDX["kl"], nm=1), rc_d)
    c = run_config(t, rc_d)["finals"]
    assert a == c, "an intervening cell must not perturb this cell (order-independence)"
    print("  PASS  order-independent  (intervening cell leaves finals unchanged)")

    # distinct seeds genuinely differ (the stream is not degenerate)
    r0 = rng_for(rc.root, "train", 0, 0, 0, 0, 1, 0).standard_normal(4)
    r1 = rng_for(rc.root, "train", 0, 0, 0, 0, 1, 1).standard_normal(4)
    assert not np.allclose(r0, r1), "distinct seeds must give distinct streams"
    print("  PASS  distinct seeds -> distinct streams")
    print("  OK    reproducibility verified\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fast smoke + reproducibility self-check")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--root", type=int, default=ROOT_SEED)
    ap.add_argument("--n-mdp", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None, help="horizon H (layers); for the H ablation")
    ap.add_argument("--peaks", type=float, nargs="+", default=None)
    ap.add_argument("--nmc", type=int, nargs="+", default=None)
    ap.add_argument("--regimes", nargs="+", default=None, choices=list(REGIME_IDX))
    ap.add_argument("--eps-list", type=float, nargs="+", default=None,
                    help="ε ablation: sweep rollout/target ε (α held fixed via --calib-eps)")
    ap.add_argument("--calib-eps", type=float, default=None,
                    help="ε used for α calibration (default = --eps / 0.2); fixes α across the ε sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.quick:
        rc = RunConfig(root=args.root, n_mdp=1, n_seeds=4, peaks=(0.7,), nmc=(1, 4),
                       regimes=("off", "off_on", "on"), steps=60, npairs=400, on_npairs=200)
        repro_check(rc)
        run(rc, args.out or "data/tabular/_smoke", jobs=args.jobs)
        return

    rc = RunConfig(root=args.root)
    if args.n_mdp is not None:   rc.n_mdp = args.n_mdp
    if args.n_seeds is not None: rc.n_seeds = args.n_seeds
    if args.steps is not None:   rc.steps = args.steps
    if args.depth is not None:   rc.depth = args.depth
    if args.peaks is not None:   rc.peaks = tuple(args.peaks)
    if args.nmc is not None:     rc.nmc = tuple(args.nmc)
    if args.regimes is not None: rc.regimes = tuple(args.regimes)
    if args.eps_list is not None:  rc.eps_list = tuple(args.eps_list)
    if args.calib_eps is not None: rc.calib_eps = args.calib_eps
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run(rc, args.out or f"data/tabular/run_{stamp}", jobs=args.jobs)


if __name__ == "__main__":
    main()
