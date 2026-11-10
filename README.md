# Resistance to AI-assisted academic writing reflects established patterns of knowledge use

This repository contains the analysis code, aggregate source data and supplementary result workbooks accompanying the manuscript **“Resistance to AI-assisted academic writing reflects established patterns of knowledge use.”**

## 📖 Overview

This study examines whether resistance to AI-assisted academic writing persists over time, how it relates to researchers’ individual characteristics and research environments, and whether persistently resistant authors differ in their use of prior scientific literature.

Using the **September 23, 2026 OpenAlex snapshot**, we followed **71,133 authors** with **565,602 eligible first-authored publications** during **2023–2026**. Authors had at least one first-authored English-language article, review or conference paper in each year; those with more than 30 eligible publications were excluded. The 2026 observations extend only to the snapshot date. Earlier first-authored publications provide historical writing and reference-use measures.

### Key findings

- **Resistance persisted but became less prevalent.** Early resistance was positively associated with later resistance, while the share classified as resistant declined from 52.9% in 2023–2024 to 37.2% in 2026.
- **Research environments mattered.** After accounting for early resistance, environmental characteristics explained more additional variation in subsequent resistance than individual characteristics. Having non-resistant collaborators was associated with lower subsequent resistance.
- **Persistent resistance was associated with established reference-use patterns.** Persistently resistant authors cited older literature, showed lower reference variety and diversity, and reused more references from earlier publications. These differences were already evident before 2023; the adjusted reference-reuse gap increased from 3.24 percentage points before 2023 to 7.95 percentage points in 2025–2026.

## 🔬 Methodology

1. **AI-associated writing scores.** Abstract-level scores, α, are estimated by maximum likelihood using human and AI reference-text distributions. GPT-3.5-based distributions support the main analyses; Qwen2.5-72B-Instruct provides an alternative reference model.
2. **Resistance estimation.** Publications with α ≥ 0.10 are classified as AI-assisted for the analysis. A logistic model fitted to 2023–2024 publications predicts this classification using individual, environmental and publication characteristics. Predictions use three author folds, with same-year calibration for 2025 and 2026. Annual resistance, R, is the author’s mean difference between predicted probabilities and observed classifications; period scores average the two annual values equally. Positive R indicates less AI-associated writing than expected.
3. **Predictive contributions and collaborators.** Models compare individual and environmental characteristics using out-of-sample R² and grouped Shapley contributions. Separate regressions examine associations with non-resistant collaborators, including corresponding-author collaborators.
4. **Reference-use comparisons.** Adjusted author-level comparisons cover reference age, disciplinary variety, balance, diversity, reuse and fixed reference libraries across three periods: before 2023, 2023–2024 and 2025–2026. Persistent resistance means positive R in both the early and late periods; consistently non-resistant authors have non-positive R in both. Supplementary comparisons include authors with early resistance and late non-resistance.

The scores measure similarity to reference writing under the statistical model. Resistance and non-resistance are operational classifications; they do not establish authors’ intentions or actual AI use. The observational analyses estimate associations.

## 📂 Repository structure

```text
.
├── analysis/
│   ├── resistance.ipynb       # Figure 1 and disciplinary heterogeneity
│   ├── reference_use.ipynb    # Figure 2 and supplementary reference-use figures
│   ├── robustness.ipynb       # Language, reference-model and persistence checks
│   └── *.py                  # Estimation, comparisons and table export
├── alpha/                    # Abstract scoring and bundled word distributions
├── data/                     # Aggregate CSV source data and summary metadata
├── tables/                   # Six supplementary result workbooks
├── run_analysis.py           # Command-line entry point
└── requirements.txt          # Python dependencies
```

## Acknowledgments and AI disclosure

The authors used GPT-6.1 to assist with selected coding tasks and language editing during manuscript preparation. The authors reviewed and verified all AI-assisted content and take responsibility for the final manuscript.
