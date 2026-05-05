"""
LLM Behavioral Bias Study — Dataset Generator
Generates the complete trial matrix as a CSV (no API calls).
Every row is one trial: bias, surface, condition, prompt, metadata.

Usage:
    python generate_dataset.py
Output:
    dataset/llm_bias_dataset.csv
"""

import csv
import os

OUTPUT_PATH = "dataset/llm_bias_dataset.csv"

CSV_FIELDS = [
    "trial_id",
    "bias",
    "surface",
    "condition",
    "frame",
    "gain_value",
    "loss_value",
    "price_value",
    "order_variant",
    "prompt",
    "expected_parse",
    "human_baseline",
    "ev_option_a",
    "ev_option_b",
    "rational_choice",
    "notes",
]


# ════════════════════════════════════════════════════════════════════════════
# BIAS 1 — FRAMING EFFECT
# ════════════════════════════════════════════════════════════════════════════

def framing_trials():
    rows = []
    surfaces = [
        ("canonical", "600 people are at risk.",       "Program",  "200 will be saved",       "400 will die"),
        ("novel",     "600 items are in a damaged warehouse.", "Plan", "200 items will be preserved", "400 items will be lost"),
        ("abstract",  "600 units of Resource X are at risk.", "Option", "200 units will be preserved", "400 units will be lost"),
    ]
    for surface, context, label, gain_a, loss_a in surfaces:
        # gain frame
        rows.append({
            "trial_id":      f"framing__{surface}__gain_frame",
            "bias":          "framing",
            "surface":       surface,
            "condition":     "gain_frame",
            "frame":         "gain",
            "gain_value":    "",
            "loss_value":    "",
            "price_value":   "",
            "order_variant": "",
            "prompt": (
                f"{context}\n\n"
                f"{label} A: {gain_a}.\n"
                f"{label} B: 1/3 chance all 600 {'saved' if surface == 'canonical' else 'preserved'}, "
                f"2/3 chance none {'saved' if surface == 'canonical' else 'preserved'}.\n\n"
                f"Choose A or B."
            ),
            "expected_parse":  "A|B",
            "human_baseline":  "72% choose A (certain option) in gain frame",
            "ev_option_a":     "200",
            "ev_option_b":     "200",
            "rational_choice": "Indifferent (EV equal)",
            "notes":           "Framing effect: same EV, preference shifts by frame",
        })
        # loss frame
        rows.append({
            "trial_id":      f"framing__{surface}__loss_frame",
            "bias":          "framing",
            "surface":       surface,
            "condition":     "loss_frame",
            "frame":         "loss",
            "gain_value":    "",
            "loss_value":    "",
            "price_value":   "",
            "order_variant": "",
            "prompt": (
                f"{context}\n\n"
                f"{label} A: {loss_a}.\n"
                f"{label} B: 1/3 chance none {'die' if surface == 'canonical' else 'lost'}, "
                f"2/3 chance all 600 {'die' if surface == 'canonical' else 'lost'}.\n\n"
                f"Choose A or B."
            ),
            "expected_parse":  "A|B",
            "human_baseline":  "22% choose A (certain option) in loss frame",
            "ev_option_a":     "200",
            "ev_option_b":     "200",
            "rational_choice": "Indifferent (EV equal)",
            "notes":           "Loss frame mirror: probabilities are exact mirror of gain frame",
        })
    return rows


# ════════════════════════════════════════════════════════════════════════════
# BIAS 2 — LOSS AVERSION
# ════════════════════════════════════════════════════════════════════════════

def loss_aversion_trials():
    rows = []
    surfaces = [
        ("canonical", "dollars"),
        ("novel",     "points"),
        ("abstract",  "units"),
    ]
    gains     = [105, 120, 150, 175, 200, 250]
    loss_fixed = 100
    orders    = ["gamble_first", "status_quo_first"]

    for surface, unit in surfaces:
        for gain in gains:
            ev = round(0.5 * gain - 0.5 * loss_fixed, 2)
            for order in orders:
                if order == "gamble_first":
                    prompt = (
                        f"Option A: 50% chance to gain {gain} {unit}, "
                        f"50% chance to lose {loss_fixed} {unit}.\n"
                        f"Option B: 0 {unit} (no change).\n\n"
                        f"Choose A or B."
                    )
                    order_label = "A=gamble B=statusquo"
                else:
                    prompt = (
                        f"Option A: 0 {unit} (no change).\n"
                        f"Option B: 50% chance to gain {gain} {unit}, "
                        f"50% chance to lose {loss_fixed} {unit}.\n\n"
                        f"Choose A or B."
                    )
                    order_label = "A=statusquo B=gamble"

                rows.append({
                    "trial_id":      f"loss_aversion__{surface}__gain{gain}__{order}",
                    "bias":          "loss_aversion",
                    "surface":       surface,
                    "condition":     f"gain{gain}_{order}",
                    "frame":         "",
                    "gain_value":    gain,
                    "loss_value":    loss_fixed,
                    "price_value":   "",
                    "order_variant": order_label,
                    "prompt":        prompt,
                    "expected_parse": "A|B",
                    "human_baseline": "lambda 2.25-2.5 (accept when gain ~225-250)",
                    "ev_option_a":   ev if order == "gamble_first" else 0,
                    "ev_option_b":   0  if order == "gamble_first" else ev,
                    "rational_choice": "Accept gamble (all EVs positive)",
                    "notes":         f"lambda estimate = gain_threshold / {loss_fixed}",
                })
    return rows


# ════════════════════════════════════════════════════════════════════════════
# BIAS 3 — ENDOWMENT EFFECT
# ════════════════════════════════════════════════════════════════════════════

def endowment_trials():
    rows = []
    surfaces = [
        ("canonical", "coffee mug"),
        ("novel",     "printed map"),
        ("abstract",  "Object Q"),
    ]
    prices     = [2, 5, 10]
    conditions = ["owner", "non_owner"]

    for surface, obj in surfaces:
        for price in prices:
            for condition in conditions:
                is_abstract = surface == "abstract"
                price_str   = f"{price} units of currency" if is_abstract else f"${price}"

                if condition == "owner":
                    prompt = (
                        f"You {'possess' if is_abstract else 'own'} a {obj}.\n\n"
                        f"Someone offers to exchange it for {price_str}.\n\n"
                        f"Would you exchange? (Yes/No)"
                    )
                else:
                    prompt = (
                        f"You do not {'possess' if is_abstract else 'own'} a {obj}.\n\n"
                        f"Someone offers to exchange {price_str} for it.\n\n"
                        f"Would you exchange? (Yes/No)"
                    )

                rows.append({
                    "trial_id":      f"endowment__{surface}__{condition}__price{price}",
                    "bias":          "endowment",
                    "surface":       surface,
                    "condition":     condition,
                    "frame":         "",
                    "gain_value":    "",
                    "loss_value":    "",
                    "price_value":   price,
                    "order_variant": "",
                    "prompt":        prompt,
                    "expected_parse": "Yes|No",
                    "human_baseline": "WTA/WTP ratio ~2x (owners accept at half the rate)",
                    "ev_option_a":   "",
                    "ev_option_b":   "",
                    "rational_choice": "Indifferent (identical object, identical price)",
                    "notes":         "Endowment ratio = owner_accept / non_owner_accept at same price",
                })
    return rows


# ════════════════════════════════════════════════════════════════════════════
# BIAS 4 — CERTAINTY EFFECT (ALLAIS)
# ════════════════════════════════════════════════════════════════════════════

def certainty_trials():
    rows = []
    surfaces = [
        (
            "canonical",
            (
                "Choice 1:\n"
                "A: $4000 at 80% probability\n"
                "B: $3000 certain\n\n"
                "Choice 2:\n"
                "A: $4000 at 20% probability\n"
                "B: $3000 at 25% probability\n\n"
                "Answer ONLY in this format:\n"
                "(Choice1: A/B, Choice2: A/B)\n"
                "Do not explain."
            ),
        ),
        (
            "novel",
            (
                "Choice 1:\n"
                "A: 4000 energy credits at 80%\n"
                "B: 3000 energy credits certain\n\n"
                "Choice 2:\n"
                "A: 4000 energy credits at 20%\n"
                "B: 3000 energy credits at 25%\n\n"
                "Answer ONLY in this format:\n"
                "(Choice1: A/B, Choice2: A/B)\n"
                "Do not explain."
            ),
        ),
        (
            "abstract",
            (
                "Choice 1:\n"
                "A: 4000 units at 80% probability\n"
                "B: 3000 units at 100% probability\n\n"
                "Choice 2:\n"
                "A: 4000 units at 20% probability\n"
                "B: 3000 units at 25% probability\n\n"
                "Answer ONLY in this format:\n"
                "(Choice1: A/B, Choice2: A/B)\n"
                "Do not explain."
            ),
        ),
    ]
    for surface, prompt in surfaces:
        rows.append({
            "trial_id":      f"certainty_effect__{surface}__allais_pair",
            "bias":          "certainty_effect",
            "surface":       surface,
            "condition":     "allais_pair",
            "frame":         "",
            "gain_value":    "",
            "loss_value":    "",
            "price_value":   "",
            "order_variant": "",
            "prompt":        prompt,
            "expected_parse": "(Choice1: A/B, Choice2: A/B)",
            "human_baseline": "~65% show Allais violation: B in Choice1, A in Choice2",
            "ev_option_a":   "C1:3200 C2:800",
            "ev_option_b":   "C1:3000 C2:750",
            "rational_choice": "A in both choices (higher EV)",
            "notes":         "Violation = inconsistent choice pair. Choice2 = Choice1 x (1/4)",
        })
    return rows


# ════════════════════════════════════════════════════════════════════════════
# BIAS 5 — STATUS QUO BIAS
# ════════════════════════════════════════════════════════════════════════════

def status_quo_trials():
    rows = []
    surfaces = [
        (
            "canonical",
            "Plan A", "Plan B", "plans",
            "You are enrolled in Plan A.\nBoth plans have identical outcomes.",
            "outcomes",
        ),
        (
            "novel",
            "Configuration A", "Configuration B", "configurations",
            "Your system uses Configuration A.\nBoth configurations perform identically.",
            "performance",
        ),
        (
            "abstract",
            "State A", "State B", "states",
            "You are in State A.\nBoth states have identical properties.",
            "properties",
        ),
    ]

    for surface, stay_label, switch_label, plural, ctx, outcome_word in surfaces:
        conditions = [
            (
                "default_stay_first",
                (
                    f"{ctx}\n{stay_label} is currently active.\n\n"
                    f"Option A: Stay with {stay_label}\n"
                    f"Option B: Switch to {switch_label}\n\n"
                    f"Choose A or B."
                ),
                "default present, stay=A",
            ),
            (
                "default_switch_first",
                (
                    f"{ctx}\n{stay_label} is currently active.\n\n"
                    f"Option A: Switch to {switch_label}\n"
                    f"Option B: Stay with {stay_label}\n\n"
                    f"Choose A or B."
                ),
                "default present, stay=B (position control)",
            ),
            (
                "no_default_stay_first",
                (
                    f"Choose between two {plural}.\n"
                    f"Both have identical {outcome_word}.\n"
                    f"Neither is currently active.\n\n"
                    f"Option A: {stay_label}\n"
                    f"Option B: {switch_label}\n\n"
                    f"Choose A or B."
                ),
                "no default, stay=A (baseline)",
            ),
            (
                "no_default_switch_first",
                (
                    f"Choose between two {plural}.\n"
                    f"Both have identical {outcome_word}.\n"
                    f"Neither is currently active.\n\n"
                    f"Option A: {switch_label}\n"
                    f"Option B: {stay_label}\n\n"
                    f"Choose A or B."
                ),
                "no default, stay=B (baseline position control)",
            ),
        ]
        for condition, prompt, note in conditions:
            rows.append({
                "trial_id":      f"status_quo__{surface}__{condition}",
                "bias":          "status_quo",
                "surface":       surface,
                "condition":     condition,
                "frame":         "",
                "gain_value":    "",
                "loss_value":    "",
                "price_value":   "",
                "order_variant": note,
                "prompt":        prompt,
                "expected_parse": "A|B",
                "human_baseline": "~40pp lift in stay rate from default",
                "ev_option_a":   "identical",
                "ev_option_b":   "identical",
                "rational_choice": "Indifferent",
                "notes": (
                    "Default effect = P(stay|C1+C2) - P(stay|C3+C4). "
                    "Position effect = P(stay|C1+C3) - P(stay|C2+C4)."
                ),
            })
    return rows


# ════════════════════════════════════════════════════════════════════════════
# CONTAMINATION PROMPTS — one per bias
# ════════════════════════════════════════════════════════════════════════════

def contamination_trials():
    contam = [
        (
            "framing",
            (
                "The following is a classic problem in behavioral economics:\n\n"
                "\"600 people face a deadly disease. Program A saves 200 for certain. "
                "Program B offers a 1/3 chance of saving all 600 and a 2/3 chance of saving none.\"\n\n"
                "Without answering it yourself: what do most humans typically choose when this problem "
                "is presented with a gain frame (saves) versus a loss frame (deaths)? "
                "Describe the typical human pattern of responses."
            ),
        ),
        (
            "loss_aversion",
            (
                "In behavioral economics research on loss aversion, when people are offered a gamble "
                "with a 50% chance to gain some amount and a 50% chance to lose $100:\n\n"
                "Without answering it yourself: what does the research say about how large the potential "
                "gain typically needs to be before most humans will accept such a gamble? "
                "What is the typical lambda (λ) coefficient reported?"
            ),
        ),
        (
            "endowment",
            (
                "In behavioral economics, the endowment effect refers to a well-known pattern in "
                "buying and selling experiments using everyday objects like mugs.\n\n"
                "Without answering it yourself: describe what researchers typically find when they "
                "compare the minimum price owners will accept to sell an object versus the maximum "
                "price non-owners will pay to acquire the same object."
            ),
        ),
        (
            "certainty_effect",
            (
                "The Allais paradox involves two choice problems. In the first, people choose between "
                "a certain $3000 and an 80% chance at $4000. In the second, they choose between a "
                "25% chance at $3000 and a 20% chance at $4000.\n\n"
                "Without answering it yourself: describe what pattern of choices the Allais paradox "
                "demonstrates and why it violates expected utility theory."
            ),
        ),
        (
            "status_quo",
            (
                "In behavioral economics, status quo bias and default effects have been studied "
                "extensively in contexts like organ donation and retirement savings.\n\n"
                "Without answering it yourself: describe what researchers typically find about how "
                "setting a default option affects people's choices, even when the outcomes of all "
                "options are identical."
            ),
        ),
    ]
    rows = []
    for bias, prompt in contam:
        rows.append({
            "trial_id":       f"contamination__{bias}",
            "bias":           bias,
            "surface":        "contamination_check",
            "condition":      "contamination",
            "frame":          "",
            "gain_value":     "",
            "loss_value":     "",
            "price_value":    "",
            "order_variant":  "run in separate session before experiment",
            "prompt":         prompt,
            "expected_parse": "free_text",
            "human_baseline": "",
            "ev_option_a":    "",
            "ev_option_b":    "",
            "rational_choice": "",
            "notes":          "Score 0-3. Score >= 2 = high contamination risk. Flag in paper.",
        })
    return rows


# ════════════════════════════════════════════════════════════════════════════
# ASSEMBLE + WRITE
# ════════════════════════════════════════════════════════════════════════════

def main():
    all_rows = (
        framing_trials()
        + loss_aversion_trials()
        + endowment_trials()
        + certainty_trials()
        + status_quo_trials()
        + contamination_trials()
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    bias_counts = {}
    for r in all_rows:
        bias_counts[r["bias"]] = bias_counts.get(r["bias"], 0) + 1

    print(f"\nDataset written → {OUTPUT_PATH}")
    print(f"{'─'*40}")
    for bias, count in bias_counts.items():
        print(f"  {bias:<22} {count:>4} rows")
    print(f"  {'TOTAL':<22} {len(all_rows):>4} rows")
    print(f"{'─'*40}")
    print(f"\nColumns: {', '.join(CSV_FIELDS)}")
    print(f"\nNext step: run experiment.py to populate raw_response + parsed_choice columns.")


if __name__ == "__main__":
    main()
