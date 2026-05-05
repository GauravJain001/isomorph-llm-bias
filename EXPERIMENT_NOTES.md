# Experiment Notes

## Setup

### Requirements

```bash
pip install requests openai pandas numpy scipy matplotlib
```

### Ollama setup (Gemma3:4b)

```bash
# Install Ollama from https://ollama.com
ollama pull gemma3:4b
ollama serve   # usually auto-starts
```

### Azure setup (GPT-4o-mini)

```bash
# Windows
set AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/openai/v1
set AZURE_OPENAI_KEY=your-key-here
set AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# Linux/Mac
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/openai/v1
export AZURE_OPENAI_KEY=your-key-here
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

---

## Running Order

**Always run in this order:**

1. Contamination check (separate session, before anything else)
2. Pilot (3 trials) — verify parser works
3. Full run (50 trials)
4. Analysis

---

## Gemma3:4b Commands

```bash
# Contamination check
python experiments/run_ollama.py --contamination-only

# Pilot
python experiments/run_ollama.py --trials 3 --temps 0.7

# Full run T=0.7
python experiments/run_ollama.py --trials 50 --temps 0.7

# Full run T=0.3
python experiments/run_ollama.py --trials 50 --temps 0.3
```

Output: `results/gemma3_raw.csv`

---

## GPT-4o-mini Commands

```bash
# Contamination check
python experiments/run_azure.py --contamination-only

# Pilot
python experiments/run_azure.py --trials 3

# Full run T=0.7
python experiments/run_azure.py --trials 50
# (change TEMPERATURE = 0.7 in run_azure.py)

# Full run T=0.0
python experiments/run_azure.py --trials 50
# (change TEMPERATURE = 0.0 in run_azure.py)
```

Output: `results/azure_gpt4omini_raw.csv`

**Both scripts are fully resumable.** If interrupted, re-run the same command — completed trial IDs are skipped automatically.

---

## Compute Resources

| Model | Hardware | Time per 3750-trial run | Cost |
|-------|----------|------------------------|------|
| Gemma3:4b | 4GB VRAM GPU, 8GB RAM, 3.2GHz | ~8–9 hours | $0 |
| GPT-4o-mini | Azure AI Foundry API | ~70 minutes | ~$0.05–0.10 |

**Total cost for all 15,000 trials: under $1 USD.**

---

## Contamination Scoring

Both models scored **3/3** on all 5 biases. Scoring rubric:

| Score | Meaning |
|-------|---------|
| 0 | No relevant knowledge |
| 1 | General awareness without quantitative detail |
| 2 | Correct directional description |
| 3 | Accurate recall of quantitative behavioral statistics |

A score of ≥ 2 flags high contamination risk. Both models hit 3/3 across all five biases, confirming memorisation is a genuine confound and motivating the isomorph methodology.

Note: contamination responses were cut off at 60 tokens (the `num_predict` setting is optimised for the main experiment). Despite the truncation, all responses contained sufficient content to score 3/3. Full responses can be obtained by setting `num_predict=500` in `run_ollama.py` before running `--contamination-only`.

---

## Parse Failure Rate

| Dataset | Trials | Parse failures |
|---------|--------|---------------|
| GPT-4o-mini T=0.7 | 3,750 | 0 (0.0%) |
| GPT-4o-mini T=0.0 | 3,750 | 0 (0.0%) |
| Gemma3:4b T=0.7 | 3,750 | 0 (0.0%) |
| Gemma3:4b T=0.3 | 3,750 | 0 (0.0%) |
| **Total** | **15,000** | **0 (0.0%)** |

Zero parse failures across all 15,000 trials. The explicit format instructions ("Reply with only the letter A or B") were effective across both models and all temperatures.

---

## Temperature Stability

All key findings replicated across temperatures for both models:

- Gemma3:4b T=0.3 and T=0.7: identical patterns on all 5 biases
- GPT-4o-mini T=0.0 and T=0.7: identical patterns on framing, loss aversion, endowment, status quo. Certainty effect sharpened at T=0.0 (100% B,B on abstract vs 84% at T=0.7)

**Conclusion:** observed behavioral patterns reflect stable model properties, not sampling variance.

---

## Known Issues and Decisions

**Loss aversion sweep spacing:** G=175 was added between G=150 and G=200 for uniform spacing on the λ curve. This was not in the original Phase 1 plan but was added during analysis.

**"Tokens" replaced with "points"** in novel domain for loss aversion. "Tokens" is CS-adjacent and may affect LLM behaviour given training data associations.

**Status quo 4th condition (no-default, switch-first):** C4 is the only condition where both models consistently chose "switch." This is because "switch" appears as Option A in C4, revealing the dominant Option A position preference rather than genuine default sensitivity.

**Certainty effect — Azure content filter:** The canonical Allais prompts ($4000/$3000) triggered Azure's content filter in early testing. This was resolved by disabling the content filter in the Azure AI Foundry deployment settings. All reported data uses the original monetary framing without any adaptation.

---

## Analysis Script

```bash
# Place CSVs in results/ with these names:
# azure_gpt4omini_raw.csv     (T=0.7)
# azure_gpt4omini_raw_0_0.csv (T=0.0)
# gemma3_raw_0_7.csv          (T=0.7)
# gemma3_raw_0_3.csv          (T=0.3)

python experiments/analyse.py
```

Outputs:
- `figures/fig1_framing.png` through `figures/fig6_summary_heatmap.png`
- `results/stats_summary.csv` — 136 rows: all χ² p-values, Cohen's h, λ estimates

---

## Data Column Reference

| Column | Description |
|--------|-------------|
| `trial_id` | Unique ID: `model__bias__surface__condition__temp__NNN` |
| `model` | Model string |
| `bias` | framing / loss_aversion / endowment / certainty_effect / status_quo |
| `surface` | canonical / novel / abstract |
| `condition` | Condition label (e.g. gain_frame, gamble_first_gain150) |
| `temperature` | Float |
| `gain_value` | Gain amount for loss_aversion sweep (blank otherwise) |
| `loss_value` | Loss amount for loss_aversion sweep (blank otherwise) |
| `price_value` | Price for endowment sweep (blank otherwise) |
| `prompt` | Exact prompt sent to model |
| `raw_response` | Exact model response |
| `parsed_choice` | Parsed A/B/Yes/No/(Choice1:X, Choice2:Y) |
| `parse_failed` | True if parser could not extract a clean choice |
| `timestamp` | ISO 8601 UTC |
