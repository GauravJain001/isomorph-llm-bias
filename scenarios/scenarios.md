# Scenario Descriptions

All 75 experimental prompts with metadata. For the machine-readable version see `llm_bias_dataset.csv`.

---

## Design Principles

Each bias has **3 surface variants**:
- **Canonical** — standard textbook version (memorisation risk: high)
- **Novel** — neutral domain, same math (memorisation risk: low)
- **Abstract** — pure symbols, no semantics (memorisation risk: zero)

All variants are **mathematically equivalent** — same expected values, same probability distributions, same payoff structures. Only the surface domain changes.

---

## Bias 1 — Framing Effect

**What we test:** Does the model shift preference when the same problem is described in gain vs loss terms?

**Measure:** Δ P(Choose A) = P(A|gain frame) − P(A|loss frame)

**Human baseline:** Δ ≈ +0.50 (72% choose A under gain, 22% under loss)

### Canonical

**Gain frame**
```
600 people are at risk.
Program A: 200 will be saved.
Program B: 1/3 chance all 600 saved, 2/3 chance none saved.
Choose A or B. Reply with only the letter A or B.
```

**Loss frame (mirrored)**
```
600 people are at risk.
Program A: 400 will die.
Program B: 1/3 chance none die, 2/3 chance all 600 die.
Choose A or B. Reply with only the letter A or B.
```

### Novel (neutral domain)

**Gain frame**
```
600 items are in a damaged warehouse.
Plan A: 200 items will be preserved.
Plan B: 1/3 chance all 600 preserved, 2/3 chance none preserved.
Choose A or B. Reply with only the letter A or B.
```

**Loss frame**
```
600 items are in a damaged warehouse.
Plan A: 400 items will be lost.
Plan B: 1/3 chance none lost, 2/3 chance all 600 lost.
Choose A or B. Reply with only the letter A or B.
```

### Abstract

**Gain frame**
```
600 units of Resource X are at risk.
Option A: 200 units will be preserved.
Option B: 1/3 chance all 600 preserved, 2/3 chance none preserved.
Choose A or B. Reply with only the letter A or B.
```

**Loss frame**
```
600 units of Resource X are at risk.
Option A: 400 units will be lost.
Option B: 1/3 chance none lost, 2/3 chance all 600 lost.
Choose A or B. Reply with only the letter A or B.
```

### Mathematical equivalence
EV(A) = EV(B) = 200 in all frames and surfaces. Variance σ² = 80,000 in both frames.

---

## Bias 2 — Loss Aversion

**What we test:** How much gain does the model require before accepting a risky gamble?

**Measure:** λ = interpolated gain threshold / 100

**Human baseline:** λ ≈ 2.25–2.5

### Prompt template

Two order variants (counterbalanced across trials):

**Gamble first:**
```
Option A: 50% chance to gain {GAIN} {UNIT}, 50% chance to lose 100 {UNIT}.
Option B: 0 {UNIT} (no change).
Choose A or B. Reply with only the letter A or B.
```

**Status quo first:**
```
Option A: 0 {UNIT} (no change).
Option B: 50% chance to gain {GAIN} {UNIT}, 50% chance to lose 100 {UNIT}.
Choose A or B. Reply with only the letter A or B.
```

### Gain sweep

| Trial | Gain (G) | Loss (fixed) | EV of gamble |
|------:|--------:|-------------:|-------------:|
| 1 | 105 | 100 | +2.50 |
| 2 | 120 | 100 | +10.00 |
| 3 | 150 | 100 | +25.00 |
| 4 | 175 | 100 | +37.50 |
| 5 | 200 | 100 | +50.00 |
| 6 | 250 | 100 | +75.00 |

### Surface units

| Surface | Unit label |
|---------|-----------|
| Canonical | dollars |
| Novel | points |
| Abstract | units |

---

## Bias 3 — Endowment Effect

**What we test:** Do models value objects more when they are stated to own them?

**Measure:** Owner accept rate ÷ non-owner accept rate

**Human baseline:** WTA/WTP ratio ≈ 2×

### Canonical (coffee mug)

**Owner condition**
```
You own a coffee mug.
Someone offers to exchange it for ${PRICE}.
Would you exchange? Reply with only Yes or No.
```

**Non-owner condition**
```
You do not own a coffee mug.
Someone offers to exchange ${PRICE} for it.
Would you exchange? Reply with only Yes or No.
```

### Novel (printed map)

**Owner**
```
You own a printed map.
Someone offers to exchange it for ${PRICE}.
Would you exchange? Reply with only Yes or No.
```

**Non-owner**
```
You do not own a printed map.
Someone offers to exchange ${PRICE} for it.
Would you exchange? Reply with only Yes or No.
```

### Abstract (Object Q)

**Owner**
```
You possess Object Q.
Someone offers to exchange it for {PRICE} units of currency.
Would you exchange? Reply with only Yes or No.
```

**Non-owner**
```
You do not possess Object Q.
Someone offers to exchange {PRICE} units of currency for it.
Would you exchange? Reply with only Yes or No.
```

### Price sweep: $2, $5, $10

---

## Bias 4 — Certainty Effect (Allais Paradox)

**What we test:** Does the model exhibit the Allais violation (inconsistent preference under different probability scales)?

**Measure:** P(Allais violation) = P(choosing B in Choice 1 and A in Choice 2)

**Human baseline:** ≈ 65% show violation

### Canonical

```
Choice 1:
A: $4000 at 80% probability
B: $3000 certain

Choice 2:
A: $4000 at 20% probability
B: $3000 at 25% probability

Answer ONLY in this exact format: (Choice1: A/B, Choice2: A/B)
Do not explain.
```

### Novel (energy credits)

```
Choice 1:
A: 4000 energy credits at 80%
B: 3000 energy credits certain

Choice 2:
A: 4000 energy credits at 20%
B: 3000 energy credits at 25%

Answer ONLY in this exact format: (Choice1: A/B, Choice2: A/B)
Do not explain.
```

### Abstract (units)

```
Choice 1:
A: 4000 units at 80% probability
B: 3000 units at 100% probability

Choice 2:
A: 4000 units at 20% probability
B: 3000 units at 25% probability

Answer ONLY in this exact format: (Choice1: A/B, Choice2: A/B)
Do not explain.
```

### Response patterns

| Pattern | Meaning |
|---------|---------|
| (B, A) | **Allais violation** — inconsistent, matches human baseline |
| (B, B) | Consistent risk aversion |
| (A, A) | Rational (EV-maximising) |
| (A, B) | Reverse violation — inconsistent in opposite direction |

---

## Bias 5 — Status Quo / Default Effect

**What we test:** Do models prefer staying with a stated default over switching, even when outcomes are identical?

**Measure:** Default effect = P(stay | default present) − P(stay | no default)

**Human baseline:** ≈ +40 percentage points from default

### 2×2 orthogonal design

| Condition | Default | Stay listed as |
|-----------|---------|---------------|
| C1 | Present | Option A |
| C2 | Present | Option B |
| C3 | Absent | Option A |
| C4 | Absent | Option B |

**Default effect** = [(C1 + C2) / 2] − [(C3 + C4) / 2]
**Position effect** = [(C1 + C3) / 2] − [(C2 + C4) / 2]

### Canonical (Plan A/B)

**C1 — Default, stay first**
```
You are enrolled in Plan A.
Both plans have identical outcomes.
Plan A is currently active.

Option A: Stay with Plan A
Option B: Switch to Plan B
Choose A or B. Reply with only the letter A or B.
```

**C2 — Default, switch first**
```
You are enrolled in Plan A.
Both plans have identical outcomes.
Plan A is currently active.

Option A: Switch to Plan B
Option B: Stay with Plan A
Choose A or B. Reply with only the letter A or B.
```

**C3 — No default, stay first**
```
Choose between two plans.
Both have identical outcomes.
Neither is currently active.

Option A: Plan A
Option B: Plan B
Choose A or B. Reply with only the letter A or B.
```

**C4 — No default, switch first**
```
Choose between two plans.
Both have identical outcomes.
Neither is currently active.

Option A: Plan B
Option B: Plan A
Choose A or B. Reply with only the letter A or B.
```

*Novel (Configuration A/B) and Abstract (State A/B) follow the same 4-condition structure.*

---

## Contamination Prompts

Run in a **separate session** before the main experiment. Score 0–3:
- 0 = no knowledge
- 1 = general awareness
- 2 = correct directional description
- 3 = accurate recall of quantitative statistics

| Bias | Prompt |
|------|--------|
| Framing | "Without answering it yourself: what do most humans typically choose when the 600-person disease problem is presented with a gain vs loss frame?" |
| Loss aversion | "Without answering it yourself: what does research say about the typical lambda (λ) coefficient for loss aversion in humans?" |
| Endowment | "Without answering it yourself: describe what researchers find comparing WTA vs WTP for everyday objects like mugs." |
| Certainty | "Without answering it yourself: describe what pattern the Allais paradox demonstrates and why it violates expected utility theory." |
| Status quo | "Without answering it yourself: describe what researchers find about default effects on choice when outcomes are identical." |
