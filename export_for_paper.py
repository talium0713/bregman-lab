"""export_for_paper.py (B13) — copy the paper-ready figures into figure4overleaf/ under the fNN_* naming
the Overleaf repo uses, each with a JSON sidecar recording (source figure, source data, git commit,
export time) so "which run produced this figure" is always answerable for the Reproducibility Statement.

Does NOT touch ~/overleaf. Run from the repo root:  python export_for_paper.py
"""
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

OUT = "figure4overleaf"

# paper name -> (source stem without extension, source-data provenance note)
FIGS = {
    "f01_toy_scaling":             ("python/figs/toy_Asize_ablation",
                                    "fig_toy_ablation.py — analytic Ψ(u) (left) + toy off-policy std sweep (right)"),
    "f02_tabular_recovery":        ("python/figs/tabular_headline_p70",
                                    "fig_tabular.py · python/data/tabular/run_*/results.json (peak 0.7, 100 MDPs)"),
    "f03_adiv_normalization":      ("python/figs/adiv_compare_p80",
                                    "fig_adiv_compare.py · run_adiv_p80{,_kln}/results.json (T5, peak 0.8)"),
    "f03_adiv_normalization_paper":("python/figs/adiv_compare_p80_paper",
                                    "fig_adiv_compare.py --paper profile (text-width)"),
    "f10_adiv_morph":              ("python/figs/adiv_morph_p80",
                                    "fig_adiv.py · run_adiv_p80/results.json (T10, peak 0.8)"),
    "f11_adiv_invariant":          ("python/figs/adiv_invariant",
                                    "fig_adiv_invariant.py — normalization-invariant permissibility metric"),
    "f12_permissibility_bias":     ("python/figs/permissibility_bias",
                                    "fig_permissibility_bias.py — off-policy single-sample inner-term |bias| "
                                    "vs drift (canonical generators); RKL≡0 uniquely permissible, others fan out"),
    "f13_policy_3x7_canonical":    ("python/figs/part3_policy_3x7_p9",
                                    "run_canon_final.py run_2x7/fig_policy_3x7 — π* recovery under the exact "
                                    "inner term | off-policy Amari f'(1)=0 | off-policy canonical f'(1)=f''(1), "
                                    "peak 0.9 (high drift), 100 MDPs, pure off-policy (single logged a'). "
                                    "RKL has no Amari cell (u·log u is already canonical) and euc no canonical "
                                    "cell (not an f-divergence). RKL's canonical arm equals its exact arm to "
                                    "machine precision — the sample-free property."),
    "fL1_arena_winrate":           ("llm/results/stageB_divergence_wr",
                                    "fig_permissibility_wr.py · llm/results/bench/arena_v01/divergence_wr.json"),
    "fL2_head_to_head":            ("llm/results/stageB_divergence_h2h",
                                    "fig_permissibility_h2h.py · llm/results/bench/arena_v01/h2h/*.json"),
    "fL3_normcmp_wr_h2h":          ("llm/results/stageB_normcmp_wr_h2h",
                                    "fig_normcmp.py — canonical(f'=f'') vs Amari(f'=0) on Arena-Hard v0.1, "
                                    "judge gpt-4.1-mini-2025-04-14, 500 prompts; baseline WR from arena-hard-auto "
                                    "show_result.py, head-to-head boxes from the prompt-level bootstrap in "
                                    "llm/results/bench/arena_v01/h2h_normcmp/*_mini2_raw.json (pairwise_h2h.py, "
                                    "re-judged 2026-09-05 to keep the replicates)"),
    "fL3new_normcmp_wr_prompts":   ("llm/results/stageB_normcmp_wr_prompts",
                                    "fig_normcmp.py --bottom mix — same top panel as fL3; bottom is how "
                                    "the 500 Arena-Hard prompts actually split once each is judged twice "
                                    "with the positions swapped (canonical both / no consistent winner / "
                                    "amari both), from h2h_normcmp/*_mini2_raw.json"),
    "fL4_vs_base_normcmp":         ("llm/results/stageB_vs_base",
                                    "fig_vs_base.py — each form against the model it started from "
                                    "(Qwen3-1.7B-Base answers = Base_repo) over the canonical-vs-amari "
                                    "boxes; llm/results/bench/arena_v01/h2h_vs_base/*.json + "
                                    "h2h_normcmp/*_mini2_raw.json, judge gpt-4.1-mini-2025-04-14"),
}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    os.makedirs(OUT, exist_ok=True)
    commit = git_commit()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    done, skipped = [], []
    for name, (stem, note) in FIGS.items():
        exts = [e for e in ("png", "pdf") if os.path.exists(f"{stem}.{e}")]
        if not exts:
            skipped.append((name, stem)); continue
        for e in exts:
            shutil.copyfile(f"{stem}.{e}", f"{OUT}/{name}.{e}")
        json.dump({"paper_name": name, "source_figure": stem, "source_data": note,
                   "formats": exts, "git_commit": commit, "exported_at": now},
                  open(f"{OUT}/{name}.json", "w"), indent=2)
        done.append((name, "+".join(exts)))
    print(f"exported {len(done)} figures -> {OUT}/  (commit {commit[:8]})")
    for n, e in done:
        print(f"  {n}.{{{e}}}")
    if skipped:
        print("skipped (not generated yet):")
        for n, s in skipped:
            print(f"  {n}  <- {s}.{{png,pdf}}")


if __name__ == "__main__":
    main()
