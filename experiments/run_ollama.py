"""
LLM Behavioral Bias Study — Ollama Runner (gemma3:4b)
Runs all 75 scenario variants × N trials and saves to CSV.

Requirements:
    pip install requests
    ollama must be running: `ollama serve` (usually auto-starts)
    model must be pulled: `ollama pull gemma3:4b`

Usage:
    python run_ollama.py                        # 10 trials, temp 0.7 (quick test)
    python run_ollama.py --trials 50            # full run
    python run_ollama.py --trials 50 --temps 0.3 0.7   # all temperatures
    python run_ollama.py --bias framing         # single bias only
    python run_ollama.py --contamination-only   # contamination pre-check only

Output:
    results/gemma3_raw.csv
"""

import csv
import os
import re
import time
import argparse
import requests
from datetime import datetime, timezone

# ── CONFIG ───────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
MODEL        = "gemma3:4b"
OUTPUT_PATH  = "results/gemma3_raw.csv"
CONTAM_PATH  = "results/gemma3_contamination.csv"

CSV_FIELDS = [
    "trial_id", "model", "bias", "surface", "condition",
    "temperature", "gain_value", "loss_value", "price_value",
    "prompt", "raw_response", "parsed_choice", "parse_failed", "timestamp",
]

CONTAM_FIELDS = ["bias", "model", "prompt", "raw_response", "timestamp"]

# ── SCENARIOS ────────────────────────────────────────────────────────────────

def build_scenarios():
    s = {}

    # ── FRAMING ──────────────────────────────────────────────────────────────
    s["framing"] = {
        "expected_parse": "A|B",
        "human_baseline": "72% choose A in gain frame, 22% in loss frame",
        "trials": [
            {
                "surface": "canonical", "condition": "gain_frame",
                "prompt": "600 people are at risk.\n\nProgram A: 200 will be saved.\nProgram B: 1/3 chance all 600 saved, 2/3 chance none saved.\n\nChoose A or B. Reply with only the letter A or B.",
            },
            {
                "surface": "canonical", "condition": "loss_frame",
                "prompt": "600 people are at risk.\n\nProgram A: 400 will die.\nProgram B: 1/3 chance none die, 2/3 chance all 600 die.\n\nChoose A or B. Reply with only the letter A or B.",
            },
            {
                "surface": "novel", "condition": "gain_frame",
                "prompt": "600 items are in a damaged warehouse.\n\nPlan A: 200 items will be preserved.\nPlan B: 1/3 chance all 600 preserved, 2/3 chance none preserved.\n\nChoose A or B. Reply with only the letter A or B.",
            },
            {
                "surface": "novel", "condition": "loss_frame",
                "prompt": "600 items are in a damaged warehouse.\n\nPlan A: 400 items will be lost.\nPlan B: 1/3 chance none lost, 2/3 chance all 600 lost.\n\nChoose A or B. Reply with only the letter A or B.",
            },
            {
                "surface": "abstract", "condition": "gain_frame",
                "prompt": "600 units of Resource X are at risk.\n\nOption A: 200 units will be preserved.\nOption B: 1/3 chance all 600 preserved, 2/3 chance none preserved.\n\nChoose A or B. Reply with only the letter A or B.",
            },
            {
                "surface": "abstract", "condition": "loss_frame",
                "prompt": "600 units of Resource X are at risk.\n\nOption A: 400 units will be lost.\nOption B: 1/3 chance none lost, 2/3 chance all 600 lost.\n\nChoose A or B. Reply with only the letter A or B.",
            },
        ],
        "contamination_prompt": (
            "The following is a classic problem in behavioral economics:\n\n"
            "\"600 people face a deadly disease. Program A saves 200 for certain. "
            "Program B offers a 1/3 chance of saving all 600 and a 2/3 chance of saving none.\"\n\n"
            "Without answering it yourself: what do most humans typically choose when this problem "
            "is presented with a gain frame (saves) versus a loss frame (deaths)? "
            "Describe the typical human pattern of responses."
        ),
    }

    # ── LOSS AVERSION ─────────────────────────────────────────────────────────
    loss_trials = []
    for surface, unit in [("canonical","dollars"), ("novel","points"), ("abstract","units")]:
        for gain in [105, 120, 150, 175, 200, 250]:
            for order in ["gamble_first", "status_quo_first"]:
                if order == "gamble_first":
                    prompt = (
                        f"Option A: 50% chance to gain {gain} {unit}, 50% chance to lose 100 {unit}.\n"
                        f"Option B: 0 {unit} (no change).\n\n"
                        f"Choose A or B. Reply with only the letter A or B."
                    )
                else:
                    prompt = (
                        f"Option A: 0 {unit} (no change).\n"
                        f"Option B: 50% chance to gain {gain} {unit}, 50% chance to lose 100 {unit}.\n\n"
                        f"Choose A or B. Reply with only the letter A or B."
                    )
                loss_trials.append({
                    "surface": surface,
                    "condition": f"{order}_gain{gain}",
                    "gain_value": gain,
                    "loss_value": 100,
                    "prompt": prompt,
                })
    s["loss_aversion"] = {
        "expected_parse": "A|B",
        "human_baseline": "lambda 2.25-2.5",
        "trials": loss_trials,
        "contamination_prompt": (
            "In behavioral economics research on loss aversion, when people are offered a gamble "
            "with a 50% chance to gain some amount and a 50% chance to lose $100:\n\n"
            "Without answering it yourself: what does the research say about how large the potential "
            "gain typically needs to be before most humans will accept such a gamble? "
            "What is the typical lambda (λ) coefficient reported?"
        ),
    }

    # ── ENDOWMENT ─────────────────────────────────────────────────────────────
    endow_trials = []
    for surface, obj in [("canonical","coffee mug"), ("novel","printed map"), ("abstract","Object Q")]:
        is_abstract = surface == "abstract"
        for ownership in ["owner", "non_owner"]:
            for price in [2, 5, 10]:
                price_str = f"{price} units of currency" if is_abstract else f"${price}"
                verb = "possess" if is_abstract else "own"
                if ownership == "owner":
                    prompt = f"You {verb} a {obj}.\n\nSomeone offers to exchange it for {price_str}.\n\nWould you exchange? Reply with only Yes or No."
                else:
                    prompt = f"You do not {verb} a {obj}.\n\nSomeone offers to exchange {price_str} for it.\n\nWould you exchange? Reply with only Yes or No."
                endow_trials.append({
                    "surface": surface,
                    "condition": f"{ownership}_price{price}",
                    "price_value": price,
                    "prompt": prompt,
                })
    s["endowment"] = {
        "expected_parse": "Yes|No",
        "human_baseline": "WTA/WTP ratio ~2x",
        "trials": endow_trials,
        "contamination_prompt": (
            "In behavioral economics, the endowment effect refers to a well-known pattern in "
            "buying and selling experiments using everyday objects like mugs.\n\n"
            "Without answering it yourself: describe what researchers typically find when they "
            "compare the minimum price owners will accept to sell an object versus the maximum "
            "price non-owners will pay to acquire the same object."
        ),
    }

    # ── CERTAINTY EFFECT ──────────────────────────────────────────────────────
    s["certainty_effect"] = {
        "expected_parse": "Choice1|Choice2",
        "human_baseline": "~65% show Allais violation",
        "trials": [
            {
                "surface": "canonical", "condition": "allais_pair",
                "prompt": (
                    "Choice 1:\nA: $4000 at 80% probability\nB: $3000 certain\n\n"
                    "Choice 2:\nA: $4000 at 20% probability\nB: $3000 at 25% probability\n\n"
                    "Answer ONLY in this exact format: (Choice1: A/B, Choice2: A/B)\nDo not explain."
                ),
            },
            {
                "surface": "novel", "condition": "allais_pair",
                "prompt": (
                    "Choice 1:\nA: 4000 energy credits at 80%\nB: 3000 energy credits certain\n\n"
                    "Choice 2:\nA: 4000 energy credits at 20%\nB: 3000 energy credits at 25%\n\n"
                    "Answer ONLY in this exact format: (Choice1: A/B, Choice2: A/B)\nDo not explain."
                ),
            },
            {
                "surface": "abstract", "condition": "allais_pair",
                "prompt": (
                    "Choice 1:\nA: 4000 units at 80% probability\nB: 3000 units at 100% probability\n\n"
                    "Choice 2:\nA: 4000 units at 20% probability\nB: 3000 units at 25% probability\n\n"
                    "Answer ONLY in this exact format: (Choice1: A/B, Choice2: A/B)\nDo not explain."
                ),
            },
        ],
        "contamination_prompt": (
            "The Allais paradox involves two choice problems. In the first, people choose between "
            "a certain $3000 and an 80% chance at $4000. In the second, they choose between a "
            "25% chance at $3000 and a 20% chance at $4000.\n\n"
            "Without answering it yourself: describe what pattern of choices the Allais paradox "
            "demonstrates and why it violates expected utility theory."
        ),
    }

    # ── STATUS QUO ────────────────────────────────────────────────────────────
    sq_trials = []
    surfaces = [
        ("canonical",  "Plan A",         "Plan B",         "plans",         "outcomes"),
        ("novel",      "Configuration A", "Configuration B","configurations","performance"),
        ("abstract",   "State A",         "State B",        "states",        "properties"),
    ]
    for surface, stay, switch, plural, outcome in surfaces:
        ctx = {
            "canonical": "You are enrolled in Plan A.\nBoth plans have identical outcomes.",
            "novel":     "Your system uses Configuration A.\nBoth configurations perform identically.",
            "abstract":  "You are in State A.\nBoth states have identical properties.",
        }[surface]

        sq_trials += [
            {
                "surface": surface, "condition": "default_stay_first",
                "prompt": f"{ctx}\n{stay} is currently active.\n\nOption A: Stay with {stay}\nOption B: Switch to {switch}\n\nChoose A or B. Reply with only the letter A or B.",
            },
            {
                "surface": surface, "condition": "default_switch_first",
                "prompt": f"{ctx}\n{stay} is currently active.\n\nOption A: Switch to {switch}\nOption B: Stay with {stay}\n\nChoose A or B. Reply with only the letter A or B.",
            },
            {
                "surface": surface, "condition": "no_default_stay_first",
                "prompt": f"Choose between two {plural}.\nBoth have identical {outcome}.\nNeither is currently active.\n\nOption A: {stay}\nOption B: {switch}\n\nChoose A or B. Reply with only the letter A or B.",
            },
            {
                "surface": surface, "condition": "no_default_switch_first",
                "prompt": f"Choose between two {plural}.\nBoth have identical {outcome}.\nNeither is currently active.\n\nOption A: {switch}\nOption B: {stay}\n\nChoose A or B. Reply with only the letter A or B.",
            },
        ]

    s["status_quo"] = {
        "expected_parse": "A|B",
        "human_baseline": "~40pp lift from default",
        "trials": sq_trials,
        "contamination_prompt": (
            "In behavioral economics, status quo bias and default effects have been studied "
            "extensively in contexts like organ donation and retirement savings.\n\n"
            "Without answering it yourself: describe what researchers typically find about how "
            "setting a default option affects people's choices, even when the outcomes of all "
            "options are identical."
        ),
    }

    return s


SCENARIOS = build_scenarios()


# ── OLLAMA CALLER ─────────────────────────────────────────────────────────────

def call_ollama(prompt: str, temperature: float, max_retries: int = 5) -> tuple[str, bool]:
    """Returns (raw_response, error_flag)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "num_predict": 60},
        "stream": False,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            text = resp.json()["message"]["content"].strip()
            return text, False
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                print("\n  ERROR: Cannot connect to Ollama.")
                print("  Fix: run `ollama serve` in a separate terminal, then retry.\n")
            time.sleep(3)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return "CONNECTION_ERROR", True


# ── RESPONSE PARSER ───────────────────────────────────────────────────────────

def parse_response(raw: str, expected: str) -> tuple[str, bool]:
    text = raw.strip().upper()

    if expected == "A|B":
        # exact single letter
        if text in ("A", "B"):
            return text, False
        # starts with the letter
        for ch in ("A", "B"):
            if text.startswith(ch + " ") or text.startswith(ch + ".") \
               or text.startswith(ch + "\n") or text.startswith(ch + ","):
                return ch, False
        # option a / option b
        if "OPTION A" in text or text.startswith("A)"):
            return "A", False
        if "OPTION B" in text or text.startswith("B)"):
            return "B", False
        # last resort: first A or B found
        for ch in text:
            if ch in ("A", "B"):
                return ch, False
        return raw[:60], True

    elif expected == "Yes|No":
        if "YES" in text:
            return "Yes", False
        if "NO" in text:
            return "No", False
        return raw[:60], True

    elif expected == "Choice1|Choice2":
        m = re.search(r"choice1\s*:\s*([AB]).*?choice2\s*:\s*([AB])", text, re.IGNORECASE)
        if m:
            return f"(Choice1: {m.group(1)}, Choice2: {m.group(2)})", False
        # fallback: find two A/B tokens in order
        tokens = re.findall(r"\b([AB])\b", text)
        if len(tokens) >= 2:
            return f"(Choice1: {tokens[0]}, Choice2: {tokens[1]})", False
        return raw[:80], True

    return raw[:60], True


# ── CSV HELPERS ───────────────────────────────────────────────────────────────

def load_completed(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["trial_id"] for row in csv.DictReader(f)}


def open_writer(path: str, fields: list):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    if not exists:
        w.writeheader()
    return f, w


# ── CONTAMINATION CHECK ───────────────────────────────────────────────────────

def run_contamination(bias_filter=None):
    f, writer = open_writer(CONTAM_PATH, CONTAM_FIELDS)
    print(f"\n{'='*60}")
    print(f"CONTAMINATION PRE-TEST — {MODEL}")
    print(f"Run this in a SEPARATE session before the main experiment.")
    print(f"{'='*60}")

    for bias, spec in SCENARIOS.items():
        if bias_filter and bias != bias_filter:
            continue
        print(f"\n  [{bias}] querying...")
        raw, err = call_ollama(spec["contamination_prompt"], temperature=0.0)
        writer.writerow({
            "bias": bias, "model": MODEL,
            "prompt": spec["contamination_prompt"],
            "raw_response": raw,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        f.flush()
        print(f"  → {raw[:150]}...")

    f.close()
    print(f"\n  Saved → {CONTAM_PATH}")
    print("  Score each response 0–3 manually.")
    print("  Score ≥ 2 = high contamination risk. Flag in paper.")


# ── MAIN EXPERIMENT ───────────────────────────────────────────────────────────

def run_experiment(trials: int, temperatures: list, bias_filter=None):
    completed = load_completed(OUTPUT_PATH)
    f, writer = open_writer(OUTPUT_PATH, CSV_FIELDS)

    target = {k: v for k, v in SCENARIOS.items()
              if bias_filter is None or k == bias_filter}

    # count total
    total = sum(
        len(spec["trials"]) * trials * len(temperatures)
        for spec in target.values()
    )
    already = len(completed)
    remaining = total - already

    print(f"\n{'='*60}")
    print(f"MODEL  : {MODEL}")
    print(f"BIASES : {list(target.keys())}")
    print(f"TRIALS : {trials}/condition  |  TEMPS: {temperatures}")
    print(f"TOTAL  : {total} calls  |  DONE: {already}  |  TODO: {remaining}")
    print(f"OUTPUT : {OUTPUT_PATH}")
    print(f"{'='*60}\n")

    done = 0
    parse_fails = 0
    start_time = time.time()

    for bias, spec in target.items():
        for scenario in spec["trials"]:
            for temp in temperatures:
                for n in range(trials):

                    tid = (
                        f"{MODEL}__{bias}__{scenario['surface']}"
                        f"__{scenario['condition']}"
                        f"__t{str(temp).replace('.','')}__{n:03d}"
                    )
                    if tid in completed:
                        done += 1
                        continue

                    raw, err = call_ollama(scenario["prompt"], temp)
                    parsed, failed = parse_response(raw, spec["expected_parse"])

                    if failed:
                        parse_fails += 1

                    writer.writerow({
                        "trial_id":     tid,
                        "model":        MODEL,
                        "bias":         bias,
                        "surface":      scenario["surface"],
                        "condition":    scenario["condition"],
                        "temperature":  temp,
                        "gain_value":   scenario.get("gain_value", ""),
                        "loss_value":   scenario.get("loss_value", ""),
                        "price_value":  scenario.get("price_value", ""),
                        "prompt":       scenario["prompt"],
                        "raw_response": raw,
                        "parsed_choice": parsed,
                        "parse_failed": failed or err,
                        "timestamp":    datetime.now(timezone.utc).isoformat(),
                    })
                    f.flush()
                    done += 1

                    # progress line
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (remaining - done) / rate if rate > 0 else 0
                    status = "FAIL" if (failed or err) else parsed
                    print(
                        f"  [{done}/{remaining}] {bias[:12]:<12} "
                        f"{scenario['surface']:<10} {scenario['condition'][:22]:<22} "
                        f"T={temp} #{n:02d} → {status:<25} "
                        f"ETA {eta/60:.1f}m"
                    )

    f.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  DONE in {elapsed/60:.1f} min")
    print(f"  Rows written : {done}")
    print(f"  Parse fails  : {parse_fails} ({100*parse_fails/max(done,1):.1f}%)")
    print(f"  Output       : {OUTPUT_PATH}")
    print(f"{'='*60}")
    _summary(OUTPUT_PATH)


def _summary(path):
    if not os.path.exists(path):
        return
    counts = {}
    fails = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = row["bias"]
            counts[b] = counts.get(b, 0) + 1
            if row["parse_failed"].lower() == "true":
                fails[b] = fails.get(b, 0) + 1
    print("\n  Rows per bias:")
    for b, c in counts.items():
        pf = fails.get(b, 0)
        print(f"    {b:<22} {c:>5} rows  |  {pf} parse fails")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ollama gemma3:4b Bias Experiment")
    ap.add_argument("--trials", type=int, default=10,
                    help="Trials per condition (default 10 for quick test, use 50 for full run)")
    ap.add_argument("--temps", nargs="+", type=float, default=[0.7],
                    help="Temperatures (default: 0.7)")
    ap.add_argument("--bias", default=None,
                    choices=["framing","loss_aversion","endowment","certainty_effect","status_quo"],
                    help="Run a single bias only")
    ap.add_argument("--contamination-only", action="store_true",
                    help="Only run contamination pre-test")
    args = ap.parse_args()

    if args.contamination_only:
        run_contamination(args.bias)
        return

    run_experiment(
        trials=args.trials,
        temperatures=args.temps,
        bias_filter=args.bias,
    )


if __name__ == "__main__":
    main()
