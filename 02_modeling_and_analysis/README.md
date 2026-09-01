# 02_modeling_and_analysis/

Modeling, statistical testing, and audit scripts run on the final catalog
produced by `01_data_preparation/`. This directory was renamed from
`data_audit/` (same 29 scripts + worker/driver siblings, filenames unchanged
-- they were already clear, numbered, and load-bearing).

## `01`, `02` -- audit-trail records, not rerun steps

`01_full_data_audit.py` and `02_feature_relationships.py` document the
manual label/coordinate fixes and feature sanity checks that informed the
current dataset. Nothing downstream reads their output; they explain *why*
the catalog looks the way it does, not a step to rerun.

## `03`-`33` -- live pipeline, run in numeric order

Every other script (including the `09b/09c/09d` and `16b/16c` worker/driver
sets) feeds either a table/number cited in Paper 1
(`report/geosite_ai_section_2026.tex`), Paper 2
(`report/geosite_ai_section_2026_paper2_regional.tex`), or the map markers
in `03_report_generation/`. Run them in filename order -- later scripts
depend on JSON/CSV outputs written by earlier ones under `results/`.

`30`-`33` were added for the N=1{,}662 update (third labeling batch,
2026-09-01): model-family comparison extended to the Domain/Infra feature
sets (`30`, `31`), the deployed national models' (GP+Infra Difficult,
Tree+Infra Easy) per-class precision/recall (`32`), and a one-off check of
the Oriental region's 4 labeled sites against those same national models,
used only for the mosaic map's Oriental fallback (`33`).

Two scripts that were part of the original `data_audit/09`/`11` numbering
were archived, not kept here (superseded by their own rewrites -- see
`archive/superseded_scripts/README.md`):
- `09_osm_routing_distance.py` -> superseded by the `09b/09c/09d` subprocess-
  isolated rewrite before the original completed.
- `11_routing_mcnemar_regional.py` -> superseded by `12_routing_lro_mcnemar.py`
  (wrong CV protocol in `11`, gave a misleading null result).

`26_paper2_mcnemar.py`'s output is computed and correct but deliberately
excluded from Paper 2's headline numbers (McNemar significance gating was
opted out for the regional paper -- see the tex preamble). Not dead, just
not cited.
