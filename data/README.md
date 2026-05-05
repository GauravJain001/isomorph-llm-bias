# Data

## Raw Response Files

| File | Model | Temperature | Trials | Notes |
|------|-------|-------------|--------|-------|
| `raw/gpt4omini_t0.7.csv` | GPT-4o-mini | 0.7 | 3,750 | Primary run |
| `raw/gpt4omini_t0.0.csv` | GPT-4o-mini | 0.0 | 3,750 | Deterministic baseline |
| `raw/gemma3_t0.7.csv` | Gemma3:4b | 0.7 | 3,750 | Primary run |
| `raw/gemma3_t0.3.csv` | Gemma3:4b | 0.3 | 3,750 | Lower temperature run |

**Total: 15,000 trials. Parse failure rate: 0.0% across all files.**

## Contamination Files

| File | Model | Biases tested |
|------|-------|---------------|
| `contamination/gpt4omini_contamination.csv` | GPT-4o-mini | 5 (all) |
| `contamination/gemma3_contamination.csv` | Gemma3:4b | 5 (all) |

Both models scored 3/3 on all five biases. See `EXPERIMENT_NOTES.md` for scoring rubric.

## Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `trial_id` | string | Unique: `model__bias__surface__condition__tTEMP__NNN` |
| `model` | string | `gpt-4o-mini` or `gemma3:4b` |
| `bias` | string | `framing` / `loss_aversion` / `endowment` / `certainty_effect` / `status_quo` |
| `surface` | string | `canonical` / `novel` / `abstract` |
| `condition` | string | Full condition label |
| `temperature` | float | Sampling temperature used |
| `gain_value` | float\|blank | Gain amount (loss_aversion only) |
| `loss_value` | float\|blank | Loss amount (loss_aversion only) |
| `price_value` | float\|blank | Price (endowment only) |
| `prompt` | string | Exact prompt sent |
| `raw_response` | string | Exact model response |
| `parsed_choice` | string | Parsed: A, B, Yes, No, or (Choice1: X, Choice2: Y) |
| `parse_failed` | bool | True if automatic parsing failed |
| `timestamp` | ISO8601 | UTC timestamp of API call |

## Bias × Surface × Condition Counts

| Bias | Variants | Trials/variant | Total |
|------|----------|----------------|-------|
| Framing | 6 (3 surf × 2 frames) | 50 | 300 |
| Loss aversion | 36 (3 surf × 6 gains × 2 orders) | 50 | 1,800 |
| Endowment | 18 (3 surf × 2 ownership × 3 prices) | 50 | 900 |
| Certainty | 3 (3 surf × 1 condition) | 50 | 150 |
| Status quo | 12 (3 surf × 4 conditions) | 50 | 600 |
| **Total per model/temp** | **75** | | **3,750** |
