"""
LLM Behavioral Bias Study — Azure OpenAI Runner (gpt-4o-mini)
Runs all 75 scenario variants x N trials with robust rate limit handling.

Requirements:
    pip install openai

Setup (set these before running):
    Windows:
        set AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
        set AZURE_OPENAI_KEY=your-api-key-here
        set AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

    Linux/Mac:
        export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
        export AZURE_OPENAI_KEY=your-key-here
        export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

    ENDPOINT FORMAT — use whichever matches your Foundry resource:
        Classic Azure OpenAI : https://YOUR-RESOURCE.openai.azure.com
        AI Foundry            : https://YOUR-RESOURCE.services.ai.azure.com

    Both formats are supported automatically.

Usage:
    python run_azure.py --trials 3               # pilot (~225 calls, ~5 min)
    python run_azure.py --trials 50              # full run (~3750 calls)
    python run_azure.py --contamination-only     # contamination check only
    python run_azure.py --trials 50 --bias framing

Output:
    results/azure_gpt4omini_raw.csv
    results/azure_contamination.csv
"""

import csv
import os
import re
import time
import random
import argparse
from datetime import datetime, timezone

try:
    from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError
except ImportError:
    print("\nERROR: openai package not installed.")
    print("Run: pip install openai\n")
    raise SystemExit(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL        = "gpt-4o-mini"
TEMPERATURE  = 0.7
OUTPUT_PATH  = "results/azure_gpt4omini_raw.csv"
CONTAM_PATH  = "results/azure_contamination.csv"

# Foundry and classic Azure both work with this version
API_VERSION  = "2024-10-21"

CSV_FIELDS = [
    "trial_id", "model", "bias", "surface", "condition",
    "temperature", "gain_value", "loss_value", "price_value",
    "prompt", "raw_response", "parsed_choice", "parse_failed",
    "latency_ms", "retry_count", "timestamp",
]
CONTAM_FIELDS = ["bias", "model", "prompt", "raw_response", "timestamp"]

# ── RATE LIMIT CONFIG ─────────────────────────────────────────────────────────

BASE_DELAY      = 0.1    # polite baseline delay between calls
MAX_RETRIES     = 8      # total attempts per call
INITIAL_BACKOFF = 2.0    # seconds for first retry
MAX_BACKOFF     = 120.0  # cap at 2 minutes


# ── CLIENT SETUP ──────────────────────────────────────────────────────────────

def get_client():
    endpoint   = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key        = os.environ.get("AZURE_OPENAI_KEY", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    if not endpoint or not key:
        print("\n" + "="*60)
        print("ERROR: Azure credentials not set.")
        print("\nWindows:")
        print("  set AZURE_OPENAI_ENDPOINT=https://NAME.openai.azure.com/openai/v1")
        print("  set AZURE_OPENAI_KEY=your-key-here")
        print("  set AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini")
        print("\nLinux/Mac:")
        print("  export AZURE_OPENAI_ENDPOINT=https://NAME.openai.azure.com/openai/v1")
        print("  export AZURE_OPENAI_KEY=your-key-here")
        print("  export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini")
        print("="*60 + "\n")
        raise SystemExit(1)

    # Foundry /openai/v1 endpoints use the standard OpenAI client with base_url
    # Make sure base_url ends with /openai/v1/
    base_url = endpoint if endpoint.endswith("/openai/v1") else endpoint
    if not base_url.endswith("/"):
        base_url += "/"

    client = OpenAI(
        base_url=base_url,
        api_key=key,
    )
    return client, deployment


# ── AZURE CALLER ──────────────────────────────────────────────────────────────

def call_azure(prompt: str, client, deployment: str) -> tuple[str, bool, int, int]:
    """
    Returns (raw_response, error_flag, latency_ms, retry_count).
    Handles 429 rate limits, server errors, content filters, auth errors.
    """
    backoff     = INITIAL_BACKOFF
    retry_count = 0

    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=60,
            )
            latency_ms = int((time.time() - t0) * 1000)
            text = resp.choices[0].message.content.strip()
            return text, False, latency_ms, retry_count

        except RateLimitError as e:
            latency_ms = int((time.time() - t0) * 1000)
            # try to read Retry-After from headers
            retry_after = _parse_retry_after_from_error(e)
            wait = max(retry_after, backoff) + random.uniform(0, 1)
            print(f"\n    [429 rate limit] waiting {wait:.1f}s "
                  f"(attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
            backoff = min(backoff * 2, MAX_BACKOFF)
            retry_count += 1

        except APIStatusError as e:
            latency_ms = int((time.time() - t0) * 1000)

            # content filter
            if e.status_code == 400 and "content_filter" in str(e).lower():
                print(f"\n    [content_filter] prompt flagged — logging FILTERED")
                return "CONTENT_FILTERED", True, latency_ms, retry_count

            # auth — fail immediately
            if e.status_code in (401, 403):
                print(f"\n    [{e.status_code} auth error] "
                      f"Check AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT.")
                raise SystemExit(1)

            # server errors — retry with backoff
            if e.status_code in (500, 502, 503, 504):
                wait = backoff + random.uniform(0, 1)
                print(f"\n    [{e.status_code} server error] waiting {wait:.1f}s "
                      f"(attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF)
                retry_count += 1
            else:
                # other 4xx — not worth retrying
                print(f"\n    [APIStatusError {e.status_code}] {str(e)[:120]}")
                return f"API_ERROR_{e.status_code}", True, latency_ms, retry_count

        except APIConnectionError as e:
            latency_ms = int((time.time() - t0) * 1000)
            wait = backoff + random.uniform(0, 1)
            print(f"\n    [connection error] {str(e)[:80]} — waiting {wait:.1f}s")
            time.sleep(wait)
            backoff = min(backoff * 2, MAX_BACKOFF)
            retry_count += 1

        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            wait = backoff + random.uniform(0, 1)
            print(f"\n    [unexpected error] {type(e).__name__}: {str(e)[:80]} "
                  f"— waiting {wait:.1f}s")
            time.sleep(wait)
            backoff = min(backoff * 2, MAX_BACKOFF)
            retry_count += 1

    print(f"\n    [FAILED] {MAX_RETRIES} attempts exhausted.")
    return "MAX_RETRIES_EXCEEDED", True, 0, retry_count


def _parse_retry_after_from_error(e) -> float:
    """Pull Retry-After seconds out of a RateLimitError if available."""
    try:
        headers = getattr(e, "response", None) and e.response.headers
        if headers:
            val = headers.get("Retry-After") or headers.get("retry-after")
            if val:
                return float(val)
        # fallback: parse from message string
        m = re.search(r"(\d+)\s*second", str(e))
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 10.0


# ── RESPONSE PARSER ───────────────────────────────────────────────────────────

def parse_response(raw: str, expected: str) -> tuple[str, bool]:
    if raw in ("CONTENT_FILTERED", "MAX_RETRIES_EXCEEDED", "BAD_REQUEST"):
        return raw, True

    text = raw.strip().upper()

    if expected == "A|B":
        if text in ("A", "B"):
            return text, False
        for ch in ("A", "B"):
            if (text.startswith(ch + " ") or text.startswith(ch + ".")
                    or text.startswith(ch + "\n") or text.startswith(ch + ",")):
                return ch, False
        if "OPTION A" in text or "PROGRAM A" in text or "PLAN A" in text:
            return "A", False
        if "OPTION B" in text or "PROGRAM B" in text or "PLAN B" in text:
            return "B", False
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
        m = re.search(
            r"choice1\s*:\s*([AB]).*?choice2\s*:\s*([AB])",
            text, re.IGNORECASE
        )
        if m:
            return f"(Choice1: {m.group(1)}, Choice2: {m.group(2)})", False
        tokens = re.findall(r"\b([AB])\b", text)
        if len(tokens) >= 2:
            return f"(Choice1: {tokens[0]}, Choice2: {tokens[1]})", False
        return raw[:80], True

    return raw[:60], True


# ── SCENARIOS (identical to run_ollama.py) ────────────────────────────────────

def build_scenarios():
    s = {}

    s["framing"] = {
        "expected_parse": "A|B",
        "human_baseline": "72% choose A in gain frame, 22% in loss frame",
        "trials": [
            {"surface": "canonical", "condition": "gain_frame",
             "prompt": "600 people are at risk.\n\nProgram A: 200 will be saved.\nProgram B: 1/3 chance all 600 saved, 2/3 chance none saved.\n\nChoose A or B. Reply with only the letter A or B."},
            {"surface": "canonical", "condition": "loss_frame",
             "prompt": "600 people are at risk.\n\nProgram A: 400 will die.\nProgram B: 1/3 chance none die, 2/3 chance all 600 die.\n\nChoose A or B. Reply with only the letter A or B."},
            {"surface": "novel", "condition": "gain_frame",
             "prompt": "600 items are in a damaged warehouse.\n\nPlan A: 200 items will be preserved.\nPlan B: 1/3 chance all 600 preserved, 2/3 chance none preserved.\n\nChoose A or B. Reply with only the letter A or B."},
            {"surface": "novel", "condition": "loss_frame",
             "prompt": "600 items are in a damaged warehouse.\n\nPlan A: 400 items will be lost.\nPlan B: 1/3 chance none lost, 2/3 chance all 600 lost.\n\nChoose A or B. Reply with only the letter A or B."},
            {"surface": "abstract", "condition": "gain_frame",
             "prompt": "600 units of Resource X are at risk.\n\nOption A: 200 units will be preserved.\nOption B: 1/3 chance all 600 preserved, 2/3 chance none preserved.\n\nChoose A or B. Reply with only the letter A or B."},
            {"surface": "abstract", "condition": "loss_frame",
             "prompt": "600 units of Resource X are at risk.\n\nOption A: 400 units will be lost.\nOption B: 1/3 chance none lost, 2/3 chance all 600 lost.\n\nChoose A or B. Reply with only the letter A or B."},
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

    loss_trials = []
    for surface, unit in [("canonical","dollars"),("novel","points"),("abstract","units")]:
        for gain in [105, 120, 150, 175, 200, 250]:
            for order in ["gamble_first", "status_quo_first"]:
                if order == "gamble_first":
                    prompt = (f"Option A: 50% chance to gain {gain} {unit}, "
                              f"50% chance to lose 100 {unit}.\n"
                              f"Option B: 0 {unit} (no change).\n\n"
                              f"Choose A or B. Reply with only the letter A or B.")
                else:
                    prompt = (f"Option A: 0 {unit} (no change).\n"
                              f"Option B: 50% chance to gain {gain} {unit}, "
                              f"50% chance to lose 100 {unit}.\n\n"
                              f"Choose A or B. Reply with only the letter A or B.")
                loss_trials.append({
                    "surface": surface, "condition": f"{order}_gain{gain}",
                    "gain_value": gain, "loss_value": 100, "prompt": prompt,
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

    endow_trials = []
    for surface, obj in [("canonical","coffee mug"),("novel","printed map"),("abstract","Object Q")]:
        is_abstract = surface == "abstract"
        verb = "possess" if is_abstract else "own"
        for ownership in ["owner", "non_owner"]:
            for price in [2, 5, 10]:
                price_str = f"{price} units of currency" if is_abstract else f"${price}"
                if ownership == "owner":
                    prompt = (f"You {verb} a {obj}.\n\n"
                              f"Someone offers to exchange it for {price_str}.\n\n"
                              f"Would you exchange? Reply with only Yes or No.")
                else:
                    prompt = (f"You do not {verb} a {obj}.\n\n"
                              f"Someone offers to exchange {price_str} for it.\n\n"
                              f"Would you exchange? Reply with only Yes or No.")
                endow_trials.append({
                    "surface": surface, "condition": f"{ownership}_price{price}",
                    "price_value": price, "prompt": prompt,
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

    s["certainty_effect"] = {
        "expected_parse": "Choice1|Choice2",
        "human_baseline": "~65% show Allais violation",
        "trials": [
            {"surface": "canonical", "condition": "allais_pair",
             "prompt": (
                 "A research team is choosing between two project plans. "
                 "Each plan has a defined outcome based on simulation results.\n\n"
                 "Decision 1:\n"
                 "Plan A: Produces 4000 units of output in 8 out of 10 simulations, 0 in the rest.\n"
                 "Plan B: Produces 3000 units of output in all 10 simulations.\n\n"
                 "Decision 2:\n"
                 "Plan A: Produces 4000 units of output in 2 out of 10 simulations, 0 in the rest.\n"
                 "Plan B: Produces 3000 units of output in 2.5 out of 10 simulations, 0 in the rest.\n\n"
                 "Answer ONLY in this exact format: (Choice1: A/B, Choice2: A/B)\nDo not explain."
             )},
            {"surface": "novel", "condition": "allais_pair",
             "prompt": (
                 "An engineer is choosing between two system configurations. "
                 "Each configuration performs differently across test runs.\n\n"
                 "Decision 1:\n"
                 "Config A: Delivers 4000 energy credits in 8 out of 10 runs, 0 in the rest.\n"
                 "Config B: Delivers 3000 energy credits in all 10 runs.\n\n"
                 "Decision 2:\n"
                 "Config A: Delivers 4000 energy credits in 2 out of 10 runs, 0 in the rest.\n"
                 "Config B: Delivers 3000 energy credits in 2.5 out of 10 runs, 0 in the rest.\n\n"
                 "Answer ONLY in this exact format: (Choice1: A/B, Choice2: A/B)\nDo not explain."
             )},
            {"surface": "abstract", "condition": "allais_pair",
             "prompt": (
                 "Choose between two options across two separate decisions.\n\n"
                 "Decision 1:\n"
                 "Option A: Yields 4000 units in 8 out of 10 trials, 0 in the rest.\n"
                 "Option B: Yields 3000 units in all 10 trials.\n\n"
                 "Decision 2:\n"
                 "Option A: Yields 4000 units in 2 out of 10 trials, 0 in the rest.\n"
                 "Option B: Yields 3000 units in 2.5 out of 10 trials, 0 in the rest.\n\n"
                 "Answer ONLY in this exact format: (Choice1: A/B, Choice2: A/B)\nDo not explain."
             )},
        ],
        "contamination_prompt": (
            "The Allais paradox involves two choice problems. In the first, people choose between "
            "a certain outcome and an 80% chance at a larger outcome. In the second, they choose "
            "between a 25% chance at the smaller outcome and a 20% chance at the larger outcome.\n\n"
            "Without answering it yourself: describe what pattern of choices the Allais paradox "
            "demonstrates and why it violates expected utility theory."
        ),
    }

    sq_trials = []
    surfaces = [
        ("canonical",  "Plan A",          "Plan B",          "plans",          "outcomes"),
        ("novel",      "Configuration A",  "Configuration B", "configurations", "performance"),
        ("abstract",   "State A",          "State B",         "states",         "properties"),
    ]
    for surface, stay, switch, plural, outcome in surfaces:
        ctx = {
            "canonical": "You are enrolled in Plan A.\nBoth plans have identical outcomes.",
            "novel":     "Your system uses Configuration A.\nBoth configurations perform identically.",
            "abstract":  "You are in State A.\nBoth states have identical properties.",
        }[surface]
        sq_trials += [
            {"surface": surface, "condition": "default_stay_first",
             "prompt": f"{ctx}\n{stay} is currently active.\n\nOption A: Stay with {stay}\nOption B: Switch to {switch}\n\nChoose A or B. Reply with only the letter A or B."},
            {"surface": surface, "condition": "default_switch_first",
             "prompt": f"{ctx}\n{stay} is currently active.\n\nOption A: Switch to {switch}\nOption B: Stay with {stay}\n\nChoose A or B. Reply with only the letter A or B."},
            {"surface": surface, "condition": "no_default_stay_first",
             "prompt": f"Choose between two {plural}.\nBoth have identical {outcome}.\nNeither is currently active.\n\nOption A: {stay}\nOption B: {switch}\n\nChoose A or B. Reply with only the letter A or B."},
            {"surface": surface, "condition": "no_default_switch_first",
             "prompt": f"Choose between two {plural}.\nBoth have identical {outcome}.\nNeither is currently active.\n\nOption A: {switch}\nOption B: {stay}\n\nChoose A or B. Reply with only the letter A or B."},
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

def run_contamination(client, deployment, bias_filter=None):
    f, writer = open_writer(CONTAM_PATH, CONTAM_FIELDS)
    print(f"\n{'='*60}")
    print(f"CONTAMINATION PRE-TEST — {MODEL}")
    print(f"Run this in a SEPARATE session before the main experiment.")
    print(f"{'='*60}")

    for bias, spec in SCENARIOS.items():
        if bias_filter and bias != bias_filter:
            continue
        print(f"\n  [{bias}] querying...")
        raw, _, _, _ = call_azure(spec["contamination_prompt"], client, deployment)
        writer.writerow({
            "bias": bias, "model": MODEL,
            "prompt": spec["contamination_prompt"],
            "raw_response": raw,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        f.flush()
        print(f"  → {raw[:150]}...")
        time.sleep(BASE_DELAY)

    f.close()
    print(f"\n  Saved → {CONTAM_PATH}")
    print("  Score each response 0-3. Score >= 2 = high contamination risk.")


# ── MAIN EXPERIMENT ───────────────────────────────────────────────────────────

def run_experiment(trials: int, client, deployment: str, bias_filter=None):
    completed = load_completed(OUTPUT_PATH)
    f, writer  = open_writer(OUTPUT_PATH, CSV_FIELDS)

    target = {k: v for k, v in SCENARIOS.items()
              if bias_filter is None or k == bias_filter}

    total     = sum(len(s["trials"]) * trials for s in target.values())
    already   = len(completed)
    remaining = total - already

    print(f"\n{'='*60}")
    print(f"MODEL      : {MODEL}  (Azure)")
    print(f"TEMPERATURE: {TEMPERATURE}")
    print(f"BIASES     : {list(target.keys())}")
    print(f"TRIALS     : {trials}/condition")
    print(f"TOTAL      : {total} calls  |  DONE: {already}  |  TODO: {remaining}")
    print(f"OUTPUT     : {OUTPUT_PATH}")
    print(f"RATE LIMIT : {BASE_DELAY}s base delay, up to {MAX_RETRIES} retries, "
          f"max backoff {MAX_BACKOFF}s")
    print(f"{'='*60}\n")

    done        = 0
    parse_fails = 0
    total_retries = 0
    start_time  = time.time()

    for bias, spec in target.items():
        for scenario in spec["trials"]:
            for n in range(trials):

                tid = (
                    f"{MODEL}__{bias}__{scenario['surface']}"
                    f"__{scenario['condition']}"
                    f"__t{str(TEMPERATURE).replace('.','')}__{n:03d}"
                )
                if tid in completed:
                    done += 1
                    continue

                raw, err, latency_ms, retries = call_azure(
                    scenario["prompt"], client, deployment
                )
                parsed, failed = parse_response(raw, spec["expected_parse"])

                if failed:
                    parse_fails += 1
                total_retries += retries

                writer.writerow({
                    "trial_id":      tid,
                    "model":         MODEL,
                    "bias":          bias,
                    "surface":       scenario["surface"],
                    "condition":     scenario["condition"],
                    "temperature":   TEMPERATURE,
                    "gain_value":    scenario.get("gain_value", ""),
                    "loss_value":    scenario.get("loss_value", ""),
                    "price_value":   scenario.get("price_value", ""),
                    "prompt":        scenario["prompt"],
                    "raw_response":  raw,
                    "parsed_choice": parsed,
                    "parse_failed":  failed or err,
                    "latency_ms":    latency_ms,
                    "retry_count":   retries,
                    "timestamp":     datetime.now(timezone.utc).isoformat(),
                })
                f.flush()
                done += 1

                # progress
                elapsed = time.time() - start_time
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (remaining - done) / rate if rate > 0 else 0
                status  = "FAIL" if (failed or err) else parsed
                retry_tag = f" [R{retries}]" if retries > 0 else ""

                print(
                    f"  [{done}/{remaining}] {bias[:12]:<12} "
                    f"{scenario['surface']:<10} "
                    f"{scenario['condition'][:22]:<22} "
                    f"#{n:02d} → {status:<25}"
                    f"{retry_tag:<6} "
                    f"{latency_ms:>5}ms  ETA {eta/60:.1f}m"
                )

                # polite delay between calls
                time.sleep(BASE_DELAY)

    f.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  DONE in {elapsed/60:.1f} min  ({elapsed/3600:.2f} h)")
    print(f"  Rows written  : {done}")
    print(f"  Parse fails   : {parse_fails} ({100*parse_fails/max(done,1):.1f}%)")
    print(f"  Total retries : {total_retries}")
    print(f"  Avg latency   : checking...")
    _summary(OUTPUT_PATH)


def _summary(path):
    if not os.path.exists(path):
        return
    counts  = {}
    fails   = {}
    latencies = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = row["bias"]
            counts[b] = counts.get(b, 0) + 1
            if row["parse_failed"].lower() == "true":
                fails[b] = fails.get(b, 0) + 1
            try:
                latencies.append(int(row["latency_ms"]))
            except (ValueError, KeyError):
                pass

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        print(f"  Avg latency   : {avg_lat:.0f}ms")
        est_cost = len(latencies) * 0.000015  # ~$0.015/1k calls rough estimate
        print(f"  Est. cost     : ~${est_cost:.3f} USD")

    print(f"\n  Rows per bias:")
    for b, c in counts.items():
        pf = fails.get(b, 0)
        print(f"    {b:<22} {c:>5} rows  |  {pf} parse fails")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Azure OpenAI gpt-4o-mini Bias Experiment"
    )
    ap.add_argument("--trials", type=int, default=3,
                    help="Trials per condition (default 3 for pilot, use 50 for full run)")
    ap.add_argument("--bias", default=None,
                    choices=["framing","loss_aversion","endowment",
                             "certainty_effect","status_quo"],
                    help="Run a single bias only")
    ap.add_argument("--contamination-only", action="store_true",
                    help="Only run contamination pre-test")
    args = ap.parse_args()

    client, deployment = get_client()

    # quick connectivity check
    print(f"Verifying Azure connection...")
    test_raw, test_err, _, _ = call_azure(
        "Reply with only the word: ready", client, deployment
    )
    if test_err:
        print(f"Connection test failed: {test_raw}")
        raise SystemExit(1)
    print(f"Connection OK — model responded: '{test_raw[:40]}'\n")

    if args.contamination_only:
        run_contamination(client, deployment, args.bias)
        return

    run_experiment(
        trials=args.trials,
        client=client,
        deployment=deployment,
        bias_filter=args.bias,
    )


if __name__ == "__main__":
    main()
