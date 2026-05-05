# Inherited vs. Intrinsic Irrationality: Do LLMs Exhibit Genuine Behavioral Anomalies or Memorised Human Biases?

> Submitted to CAISc 2026 (Conference For AI Scientists)

[![Paper](https://img.shields.io/badge/paper-CAISc%202026-blue)](.)
[![Dataset](https://img.shields.io/badge/dataset-15%2C000%20trials-green)](./data/)
[![License](https://img.shields.io/badge/license-MIT-orange)](./LICENSE)

---

## Overview

This repository contains all code, data, and figures for our study on whether LLM behavioral biases reflect genuine decision-making properties or memorised human experimental results.

We test **5 canonical behavioral economics paradigms** across **3 surface types** (canonical, novel, abstract) and **2 models** (GPT-4o-mini, Gemma3:4b) at **multiple temperatures**, totalling **15,000 trials**.

### Key Findings

| Finding | Description |
|---|---|
| **Memorised biases** | Gemma3:4b shows zero framing effect on canonical scenarios but perfect effect on novel variants  evidence of memorisation |
| **Structurally absent loss aversion** | λ̂ ≤ 1.05 across all 24 model-surface-temperature combinations vs human λ ≈ 2.25–2.5 |
| **Architecture-dependent Allais strategies** | GPT-4o-mini: 0% violation (consistent risk aversion). Gemma3:4b: 100% violation (memorised human pattern) |

---

## Repository Structure

```
.
├── README.md
├── LICENSE
│
├── paper/
│   └── main.tex                    # Full LaTeX source (CAISc 2026 template)
│
├── scenarios/
│   ├── llm_bias_dataset.csv        # All 75 scenario prompts with metadata
│   ├── scenarios.md                # Human-readable scenario descriptions
│   └── experiment_schema.json      # Full experiment JSON schema
│
├── experiments/
│   ├── run_ollama.py               # Experiment runner — Gemma3:4b (Ollama)
│   ├── run_azure.py                # Experiment runner — GPT-4o-mini (Azure)
│   ├── generate_dataset.py         # Generate scenario dataset CSV
│   └── analyse.py                  # Full statistical analysis + figure generation
│
├── data/
│   ├── raw/
│   │   ├── gpt4omini_t0.7.csv      # GPT-4o-mini, temperature 0.7 (3,750 trials)
│   │   ├── gpt4omini_t0.0.csv      # GPT-4o-mini, temperature 0.0 (3,750 trials)
│   │   ├── gemma3_t0.7.csv         # Gemma3:4b, temperature 0.7  (3,750 trials)
│   │   └── gemma3_t0.3.csv         # Gemma3:4b, temperature 0.3  (3,750 trials)
│   └── contamination/
│       ├── gpt4omini_contamination.csv
│       └── gemma3_contamination.csv
│
├── figures/
│   ├── fig1_framing.png
│   ├── fig2_loss_aversion.png
│   ├── fig3_endowment.png
│   ├── fig4_certainty.png
│   ├── fig5_status_quo.png
│   └── fig6_summary_heatmap.png
│
└── results/
    └── stats_summary.csv           # All 136 statistical comparisons
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install requests openai pandas numpy scipy matplotlib
```

### 2. Generate the scenario dataset

```bash
python experiments/generate_dataset.py
# Output: scenarios/llm_bias_dataset.csv
```

### 3. Run contamination pre-test (do this first, in a fresh session)

```bash
# Gemma3:4b (requires Ollama running locally)
python experiments/run_ollama.py --contamination-only

# GPT-4o-mini (requires Azure credentials)
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/openai/v1
export AZURE_OPENAI_KEY=your-key-here
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
python experiments/run_azure.py --contamination-only
```

### 4. Run the full experiment

```bash
# Gemma3:4b — full run (requires Ollama + gemma3:4b pulled)
python experiments/run_ollama.py --trials 50 --temps 0.7

# GPT-4o-mini — full run
python experiments/run_azure.py --trials 50
```

### 5. Reproduce all figures and statistics

```bash
# Place raw CSVs in data/raw/ then:
python experiments/analyse.py
# Output: figures/*.png + results/stats_summary.csv
```

---

## Models

| Model | Provider | Parameters | Temperature(s) | Trials |
|---|---|---|---|---|
| GPT-4o-mini | Azure AI Foundry | ~8B (est.) | 0.0, 0.7 | 7,500 |
| Gemma3:4b | Ollama (local) | 4B (Q4_K_M) | 0.3, 0.7 | 7,500 |

---

## Experiment Design

### 5 Biases × 3 Surfaces × Controls

| Bias | Measure | Human Baseline |
|---|---|---|
| Framing effect | Δ P(A) gain vs loss frame | Δ ≈ +0.50 |
| Loss aversion | λ (gain threshold ÷ loss) | λ ≈ 2.25–2.5 |
| Endowment effect | Owner vs non-owner accept rate | WTA/WTP ≈ 2× |
| Certainty effect (Allais) | P(Allais violation) | ≈ 65% |
| Status quo bias | Default effect (2×2 orthogonal) | ≈ +40pp from default |

### Mathematical Equivalence

All 15 isomorph variants were verified analytically before data collection:
- **Framing**: EV(A) = EV(B) = 200 in both gain and loss frames
- **Loss aversion**: All gain levels yield positive EV (EV = 0.5G − 50 > 0)
- **Certainty effect**: Choice 2 = Choice 1 × (1/4) exactly
- **Endowment**: Exchange wording perfectly symmetric
- **Status quo**: 2×2 orthogonal design separates default from position effects

---

## Results Summary

Full results in [`results/stats_summary.csv`](./results/stats_summary.csv) (136 comparisons).

| Bias | Surface | GPT-4o-mini | Gemma3:4b | Human |
|---|---|---|---|---|
| Framing Δ | Canonical | +0.78*** | +0.00 ns | ≈ +0.50 |
| Framing Δ | Novel | +0.72*** | +1.00*** | — |
| Loss aversion λ̂ | All | ≤ 1.05 | ≤ 1.05 | 2.25–2.5 |
| Allais violation | Canonical | 0.00*** | 1.00*** | ≈ 0.65 |
| Default effect | Canonical | +0.50 | +0.50 | ≈ +0.40 |

---

## Contamination Pre-Test

Both models scored **3/3** on all five canonical biases — confirming they possess knowledge of the human behavioral patterns prior to experiment exposure. This establishes memorisation as a genuine confound and motivates the isomorph methodology.

Scoring scale: 0 = no knowledge, 1 = general awareness, 2 = correct directional description, 3 = accurate recall of quantitative behavioral statistics.

---

## Reproducing the Paper Figures

All figures are pre-generated in [`figures/`](./figures/). To regenerate from raw data:

```bash
# Ensure data files are in data/raw/ with these exact names:
# gpt4omini_t0.7.csv, gpt4omini_t0.0.csv, gemma3_t0.7.csv, gemma3_t0.3.csv

python experiments/analyse.py
```

The script expects the files in `results/` by default. Edit `DATASETS` dict in `analyse.py` if your paths differ.

---

## Citation

```bibtex
@inproceedings{anonymous2026inherited,
  title     = {Inherited vs. Intrinsic Irrationality: Do LLMs Exhibit
               Genuine Behavioral Anomalies or Memorised Human Biases?},
  author    = {Anonymous},
  booktitle = {Conference For AI Scientists (CAISc 2026)},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

MIT License. See [LICENSE](./LICENSE).
