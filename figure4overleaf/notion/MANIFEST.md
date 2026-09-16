# Notion figure assets

Images embedded in the **Experiment Writing** Notion page
(https://app.notion.com/p/3d8cc1ead0d6800ab5cbed6baae20c42).

Copies, not originals: the source of truth stays in `figure4overleaf/<paper_name>.png`,
which is itself exported from `python/figs/` or `llm/results/` by `export_for_paper.py`.
Re-export upstream first, then refresh here and re-upload to Notion.

| Notion file | paper figure | section |
| --- | --- | --- |
| `s2_toy_psi_and_scaling.png` | `f01_toy_scaling` | §2 toy env — Ψ(u) and the \|A\|-scaling of inner-term noise |
| `s3_tabular_recovery_nmc.png` | `f02_tabular_recovery` | §3 tabular — off-policy bars + Δπ vs n_mc |
| `s4_adiv_canonical_vs_amari.png` | `f03_adiv_normalization_paper` | §4 α-divergence — canonical vs Amari over α |
| `s5_arena_canonical_vs_amari.png` | `fL3_normcmp_wr_h2h` | §5 LLM — Arena-Hard win rate + head-to-head |
| `s5_arena_vs_base.png` | `fL4_vs_base_normcmp` | §5 LLM — each form against the untrained base |
| `apx_adiv_morph.png` | `f10_adiv_morph` | appendix |
| `apx_adiv_invariant.png` | `f11_adiv_invariant` | appendix |
| `apx_inner_term_bias.png` | `f12_permissibility_bias` | appendix — single-sample inner-term bias vs drift |
| `apx_policy_recovery_3x7.png` | `f13_policy_3x7_canonical` | appendix — π* recovery, 3×7 panel (exact / Amari / canonical) |
| `apx_arena_winrate_all.png` | `fL1_arena_winrate` | appendix — all 7 divergences vs baseline |
| `apx_arena_h2h_all.png` | `fL2_head_to_head` | appendix — RKL head-to-head vs each divergence |
| `apx_arena_prompt_mix.png` | `fL3new_normcmp_wr_prompts` | appendix — prompt-outcome mix variant |

## Naming
`s<section>_` for main text, `apx_` for appendix. The paper-side name is the `paper_name`
field in the matching `figure4overleaf/*.json` sidecar; keep both in sync when renaming.

## Note
`apx_inner_term_bias.png` still comes from `fig_permissibility_bias.py`. The paper no longer
uses "permissibility", so that script and its sidecar are due a rename.
