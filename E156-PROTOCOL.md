# E156 Protocol — `metaoverfit`

This repository is the source code and dashboard backing an E156 micro-paper on the [E156 Student Board](https://mahmood726-cyber.github.io/e156/students.html).

---

## `[99]` metaoverfit: Optimism-Corrected Heterogeneity Assessment for Meta-Regression

**Type:** methods  |  ESTIMAND: Optimism-corrected R-squared  
**Data:** 12,000 simulations, 501 Cochrane meta-analyses

### 156-word body

Does overfitting inflate the apparent proportion of heterogeneity explained in published meta-regressions, and can cross-validation correct this distortion? We developed the metaoverfit R package implementing leave-one-out cross-validation of between-study variance and bootstrap optimism correction, built on metafor with parallel computation and automatic risk classification. The package exports six functions covering apparent and adjusted R-squared, LOOCV-corrected estimates, bootstrap confidence intervals with boundary diagnostics, audit reports, visualization, and sample-size planning guidance. Across 12,000 simulations the median optimism was 0.18 (95% CI 0.12 to 0.25) and apparent R-squared exceeded 50 percent under null models when the studies-to-predictors ratio fell below five. Analysis of 501 Cochrane meta-analyses revealed that over 30 percent had severe or extreme overfitting risk at the corrected threshold. Automated risk categorization assigns each analysis to one of four tiers based on ratio thresholds and observed shrinkage magnitude. A limitation is that cross-validation with fewer than ten studies may itself be unstable and should be interpreted cautiously.

### Submission metadata

```
Corresponding author: Mahmood Ahmad <mahmood.ahmad2@nhs.net>
ORCID: 0000-0001-9107-3704
Affiliation: Tahir Heart Institute, Rabwah, Pakistan

Links:
  Code:      https://github.com/mahmood726-cyber/metaoverfit
  Protocol:  https://github.com/mahmood726-cyber/metaoverfit/blob/main/E156-PROTOCOL.md
  Dashboard: https://mahmood726-cyber.github.io/metaoverfit/

References (topic pack: heterogeneity / prediction interval):
  1. Higgins JPT, Thompson SG. 2002. Quantifying heterogeneity in a meta-analysis. Stat Med. 21(11):1539-1558. doi:10.1002/sim.1186
  2. IntHout J, Ioannidis JP, Rovers MM, Goeman JJ. 2016. Plea for routinely presenting prediction intervals in meta-analysis. BMJ Open. 6(7):e010247. doi:10.1136/bmjopen-2015-010247

Data availability: No patient-level data used. Analysis derived exclusively
  from publicly available aggregate records. All source identifiers are in
  the protocol document linked above.

Ethics: Not required. Study uses only publicly available aggregate data; no
  human participants; no patient-identifiable information; no individual-
  participant data. No institutional review board approval sought or required
  under standard research-ethics guidelines for secondary methodological
  research on published literature.

Funding: None.

Competing interests: MA serves on the editorial board of Synthēsis (the
  target journal); MA had no role in editorial decisions on this
  manuscript, which was handled by an independent editor of the journal.

Author contributions (CRediT):
  [STUDENT REWRITER, first author] — Writing – original draft, Writing –
    review & editing, Validation.
  [SUPERVISING FACULTY, last/senior author] — Supervision, Validation,
    Writing – review & editing.
  Mahmood Ahmad (middle author, NOT first or last) — Conceptualization,
    Methodology, Software, Data curation, Formal analysis, Resources.

AI disclosure: Computational tooling (including AI-assisted coding via
  Claude Code [Anthropic]) was used to develop analysis scripts and assist
  with data extraction. The final manuscript was human-written, reviewed,
  and approved by the author; the submitted text is not AI-generated. All
  quantitative claims were verified against source data; cross-validation
  was performed where applicable. The author retains full responsibility for
  the final content.

Preprint: Not preprinted.

Reporting checklist: PRISMA 2020 (methods-paper variant — reports on review corpus).

Target journal: ◆ Synthēsis (https://www.synthesis-medicine.org/index.php/journal)
  Section: Methods Note — submit the 156-word E156 body verbatim as the main text.
  The journal caps main text at ≤400 words; E156's 156-word, 7-sentence
  contract sits well inside that ceiling. Do NOT pad to 400 — the
  micro-paper length is the point of the format.

Manuscript license: CC-BY-4.0.
Code license: MIT.

SUBMITTED: [ ]
```


---

_Auto-generated from the workbook by `C:/E156/scripts/create_missing_protocols.py`. If something is wrong, edit `rewrite-workbook.txt` and re-run the script — it will overwrite this file via the GitHub API._