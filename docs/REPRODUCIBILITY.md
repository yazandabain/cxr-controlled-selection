# Reproducing the public outputs

This repository supports two different checks:

1. Synthetic tests and a small example exercise selection, fitting and statistics without patient data.
2. Aggregate presentation replay rebuilds the reported quantitative figures and numerical tables without models, predictions, bootstrap arrays or credentials.

Neither check reruns the original patient-level study. Cohort extraction, feature generation and model fitting require authorized private inputs that are not distributed here.

## Commands

```bash
python -m pip install -c requirements-ci.txt ".[trees,figures,test]"
python -m pip check
python -m unittest discover -s tests -v
python examples/synthetic_study.py --draws 200
python scripts/build_assets.py --out ../cxr-public-assets
python scripts/check_release.py
```

Choose a fresh output directory: existing outputs are not overwritten. Generated files stay untracked. The public inputs are in `results/aggregates`; reviewed figure exports are in `figures`; readable table exports are in `results/tables`.

The builder produces seven quantitative figure pairs (SVG/PDF) and fifteen tables (CSV/Markdown/LaTeX fragments). It records every plotted aggregate entry and derived display calculation in a numeric ledger, then verifies entries against a fresh source reread. The conceptual Figure 1 has editable draw.io source and SVG/PDF exports; edit or export it with draw.io rather than redrawing it in Python. No draw.io executable or journal style file is bundled.

Tests independently check plotted composition fractions and the AUROC pair sum, compare generated table cells with retained exports and inspect the public diagram for external or embedded image content. The retained quantitative SVGs were checked against the verified renderer before packaging. Presentation replay verifies the scientific values without requiring byte identity across platforms. PDF metadata, font libraries and heatmap encoding can affect file bytes.

## Runtime and CI

Python 3.13 is the release-test runtime. The package requires Python 3.11 or later because its numerical dependencies require that minimum, but other versions are not claimed as tested. Official package metadata supplies Python 3.13 wheels for the pinned numerical, plotting and tree dependencies on Windows and Linux. CI installs the package from a fresh checkout on Python 3.13 on both operating systems and runs the complete public checks.

The workflow uses read-only repository permissions, pinned GitHub Action commits, shared test-environment constraints and no patient data or secrets. The constraints record the verified Python 3.13 environment; platform-specific dependencies remain governed by their packages. Its badge will show a result only after the repository is pushed and CI runs. Local success is not a claim that hosted CI has already passed.

## Aggregate disclosure boundary

The public JSON contains study-level **summaries**, not study rows: group counts, descriptive statistics, performance estimates, interval endpoints and twenty per-fit group-composition summaries. Per-fit composition is a report-state count by coded class, not a patient membership list. Pair-accounting arrays are fixed four-by-four aggregate matrices, not bootstrap draws.

Private artifact hashes and run/source-code hashes were removed from the exported inputs. Scientific numbers and methodological qualifications were retained. The public aggregate copies and plot builder are separate from the unchanged internal evidence.

The [PhysioNet credentialed-data license](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/) prohibits sharing restricted data and requires care against identity disclosure in publications. The selected release outputs were reviewed as summary research results with no patient records, identifiers, linked predictions or embedded clinical images. This is a technical disclosure review, not a blanket authorization for arbitrary future derivatives.

## Rights

Ownership and authority to grant a permissive source-code license remain unconfirmed. No open-source license has been applied. See [NOTICE.md](../NOTICE.md). Dependencies are installed, not vendored, and their terms remain separate. The release has no Springer templates, pretrained checkpoints, external icon assets or copied paper text.
