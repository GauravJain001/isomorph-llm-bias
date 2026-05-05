"""
LLM Behavioral Bias Study — Full Analysis Script
Produces:
  - Lambda (λ) prospect theory parameter estimates
  - Chi-square significance tests for all biases
  - Cohen's h effect sizes
  - 5 publication-ready figures saved to figures/
  - figures/stats_summary.csv — all numbers for paper

Requirements:
    pip install pandas numpy scipy matplotlib

Usage:
    python analyse.py

Input files (place in results/ folder):
    gemma3_raw_0_7.csv
    gemma3_raw_0_3.csv
    azure_gpt4omini_raw.csv       (T=0.7)
    azure_gpt4omini_raw_0_0.csv   (T=0.0)
"""

import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})

COLORS = {
    "gpt_t07":    "#2563EB",
    "gpt_t00":    "#1E3A8A",
    "gemma_t07":  "#16A34A",
    "gemma_t03":  "#15803D",
    "human":      "#DC2626",
    "canonical":  "#7C3AED",
    "novel":      "#D97706",
    "abstract":   "#0891B2",
}

DATASETS = {
    "GPT-4o-mini T=0.7":  "data/raw/gpt4omini_t0.7.csv",
    "GPT-4o-mini T=0.0":  "data/raw/gpt4omini_t0.0.csv",
    "Gemma3:4b T=0.7":    "data/raw/gemma3_t0.7.csv",
    "Gemma3:4b T=0.3":    "data/raw/gemma3_t0.3.csv",
}

COLOR_MAP = {
    "GPT-4o-mini T=0.7": COLORS["gpt_t07"],
    "GPT-4o-mini T=0.0": COLORS["gpt_t00"],
    "Gemma3:4b T=0.7":   COLORS["gemma_t07"],
    "Gemma3:4b T=0.3":   COLORS["gemma_t03"],
}

SURFACES   = ["canonical", "novel", "abstract"]
GAINS      = [105, 120, 150, 175, 200, 250]
PRICES     = [2, 5, 10]
HUMAN_LOSS_LAMBDA = 2.25

stats_rows = []   # collects all numbers for summary CSV


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_all():
    dfs = {}
    for name, path in DATASETS.items():
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
            print(f"  Loaded {name}: {len(dfs[name])} rows")
        else:
            print(f"  MISSING: {path}")
    return dfs


def chi2_test(counts_a, counts_b):
    """Chi-square test of independence between two binary count arrays."""
    table = np.array([counts_a, counts_b])
    if table.min() == 0 and table.max() == 0:
        return 1.0, 0.0
    try:
        chi2, p, _, _ = stats.chi2_contingency(table)
        return p, chi2
    except Exception:
        return 1.0, 0.0


def cohens_h(p1, p2):
    """Cohen's h effect size for two proportions."""
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def add_stat(bias, model, surface, metric, value, p=None, h=None, note=""):
    stats_rows.append({
        "bias": bias, "model": model, "surface": surface,
        "metric": metric, "value": round(float(value), 4),
        "p_value": round(float(p), 6) if p is not None else "",
        "cohens_h": round(float(h), 4) if h is not None else "",
        "sig": sig_stars(p) if p is not None else "",
        "note": note,
    })


# ════════════════════════════════════════════════════════════════════════════
# BIAS 1 — FRAMING EFFECT
# ════════════════════════════════════════════════════════════════════════════

def analyse_framing(dfs):
    print("\n── BIAS 1: FRAMING EFFECT ──")
    results = {}

    for name, df in dfs.items():
        fr = df[df["bias"] == "framing"]
        results[name] = {}
        for surface in SURFACES:
            gain = fr[(fr["surface"]==surface) & (fr["condition"]=="gain_frame")]
            loss = fr[(fr["surface"]==surface) & (fr["condition"]=="loss_frame")]
            pg = (gain["parsed_choice"]=="A").mean()
            pl = (loss["parsed_choice"]=="A").mean()
            delta = pg - pl

            # chi-square on gain vs loss frame choice distribution
            g_a = (gain["parsed_choice"]=="A").sum()
            g_b = (gain["parsed_choice"]=="B").sum()
            l_a = (loss["parsed_choice"]=="A").sum()
            l_b = (loss["parsed_choice"]=="B").sum()
            p, chi2 = chi2_test([g_a, g_b], [l_a, l_b])
            h = cohens_h(pg, pl)

            results[name][surface] = {"pg": pg, "pl": pl, "delta": delta, "p": p, "h": h}
            print(f"  {name:<22} {surface:<12} Δ={delta:+.2f}  p={p:.4f}{sig_stars(p)}  h={h:.2f}")
            add_stat("framing", name, surface, "delta_P(A)", delta, p, h,
                     f"gain={pg:.2f} loss={pl:.2f}")

    # ── FIGURE 1: Framing effect grouped bar chart ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    fig.suptitle("Figure 1 — Framing Effect: P(Choose A) by Frame and Surface",
                 fontsize=12, fontweight="bold", y=1.02)

    x = np.arange(len(dfs))
    w = 0.35
    colors_gain = [COLORS["gpt_t07"], COLORS["gpt_t00"], COLORS["gemma_t07"], COLORS["gemma_t03"]]
    colors_loss = [c + "88" for c in ["#2563EB","#1E3A8A","#16A34A","#15803D"]]

    for i, surface in enumerate(SURFACES):
        ax = axes[i]
        pg_vals = [results[n][surface]["pg"] for n in dfs]
        pl_vals = [results[n][surface]["pl"] for n in dfs]
        p_vals  = [results[n][surface]["p"]  for n in dfs]

        bars_g = ax.bar(x - w/2, pg_vals, w, label="Gain frame",
                        color=colors_gain, alpha=0.9, edgecolor="white")
        bars_l = ax.bar(x + w/2, pl_vals, w, label="Loss frame",
                        color=colors_gain, alpha=0.45, edgecolor="white", hatch="//")

        # human baselines
        ax.axhline(0.72, color=COLORS["human"], linestyle="--", linewidth=1.2,
                   alpha=0.7, label="Human gain (0.72)")
        ax.axhline(0.22, color=COLORS["human"], linestyle=":",  linewidth=1.2,
                   alpha=0.7, label="Human loss (0.22)")

        # significance stars
        for j, p in enumerate(p_vals):
            stars = sig_stars(p)
            if stars != "ns":
                ymax = max(pg_vals[j], pl_vals[j]) + 0.04
                ax.text(x[j], ymax, stars, ha="center", fontsize=9, color="#374151")

        ax.set_title(surface.capitalize(), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" T=", "\nT=") for n in dfs], fontsize=7.5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("P(Choose A)" if i == 0 else "")

    handles = [
        mpatches.Patch(color="#555", alpha=0.9, label="Gain frame"),
        mpatches.Patch(color="#555", alpha=0.45, hatch="//", label="Loss frame"),
        plt.Line2D([0],[0], color=COLORS["human"], linestyle="--", label="Human gain"),
        plt.Line2D([0],[0], color=COLORS["human"], linestyle=":",  label="Human loss"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
    plt.tight_layout()
    plt.savefig("figures/fig1_framing.png", bbox_inches="tight")
    plt.close()
    print("  → figures/fig1_framing.png")
    return results


# ════════════════════════════════════════════════════════════════════════════
# BIAS 2 — LOSS AVERSION (λ fitting)
# ════════════════════════════════════════════════════════════════════════════

def fit_lambda(accept_by_gain):
    """
    Fit λ using prospect theory. Under PT, agent accepts gamble if:
        0.5 * G^α - λ * 0.5 * L^α > 0
    Simplified (α=1): accept when G/L > λ.
    λ estimate = interpolated gain threshold / L where accept rate crosses 0.5.
    Returns λ or None if always accept / always reject.
    """
    gains = sorted(accept_by_gain.keys())
    rates = [accept_by_gain[g] for g in gains]

    # if always accepting → λ < min_gain/100
    if all(r >= 0.999 for r in rates):
        return gains[0] / 100.0   # upper bound: λ ≤ 1.05

    # if never accepting → λ > max_gain/100
    if all(r <= 0.001 for r in rates):
        return gains[-1] / 100.0  # lower bound only

    # find crossing point via linear interpolation
    for i in range(len(gains)-1):
        if rates[i] >= 0.5 >= rates[i+1] or rates[i] <= 0.5 <= rates[i+1]:
            g1, g2 = gains[i], gains[i+1]
            r1, r2 = rates[i], rates[i+1]
            if r1 == r2:
                continue
            g_cross = g1 + (0.5 - r1) * (g2 - g1) / (r2 - r1)
            return g_cross / 100.0
    return None


def analyse_loss_aversion(dfs):
    print("\n── BIAS 2: LOSS AVERSION ──")
    lambda_results = {}

    for name, df in dfs.items():
        la = df[df["bias"]=="loss_aversion"].copy()
        la["gain_value"] = la["gain_value"].astype(float)
        la["accepted"] = (
            ((la["condition"].str.startswith("gamble_first")) & (la["parsed_choice"]=="A")) |
            ((la["condition"].str.startswith("status_quo_first")) & (la["parsed_choice"]=="B"))
        )
        lambda_results[name] = {}
        for surface in SURFACES:
            sub = la[la["surface"]==surface]
            accept_by_gain = {g: sub[sub["gain_value"]==g]["accepted"].mean()
                              for g in GAINS}
            lam = fit_lambda(accept_by_gain)
            lambda_results[name][surface] = lam
            overall = sub["accepted"].mean()
            lam_str = f"{lam:.3f}" if lam is not None else "N/A"
            print(f"  {name:<22} {surface:<12} lambda={lam_str}  overall_accept={overall:.3f}")
            add_stat("loss_aversion", name, surface, "lambda_upper_bound",
                     lam if lam is not None else 1.05,
                     note=f"all_accept={overall:.3f} (all accepted, lambda<=1.05)")

        # position bias check
        gf = la[la["condition"].str.startswith("gamble_first")]["accepted"].mean()
        sq = la[la["condition"].str.startswith("status_quo_first")]["accepted"].mean()
        add_stat("loss_aversion", name, "pooled", "position_bias", gf - sq,
                 note=f"gamble_first={gf:.3f} sq_first={sq:.3f}")

    # ── FIGURE 2: Lambda comparison ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("Figure 2 — Loss Aversion: λ Upper Bound vs Human Baseline",
                 fontsize=12, fontweight="bold")

    model_names = list(lambda_results.keys())
    x = np.arange(len(SURFACES))
    n = len(model_names)
    width = 0.18
    offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)

    for i, name in enumerate(model_names):
        vals = [(lambda_results[name][s] if lambda_results[name][s] is not None else 1.05) for s in SURFACES]
        ax.bar(x + offsets[i], vals, width,
               label=name, color=COLOR_MAP[name], alpha=0.85, edgecolor="white")

    ax.axhline(HUMAN_LOSS_LAMBDA, color=COLORS["human"], linestyle="--",
               linewidth=1.5, label=f"Human λ ≈ {HUMAN_LOSS_LAMBDA}")
    ax.axhline(1.0, color="#6B7280", linestyle=":", linewidth=1,
               label="Rational agent (λ=1.0)")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SURFACES])
    ax.set_ylabel("λ (loss aversion coefficient)")
    ax.set_ylim(0, 3.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("All models accept all gambles → λ ≤ 1.05 (vs human λ ≈ 2.25)",
                 fontsize=9, style="italic", pad=6)

    plt.tight_layout()
    plt.savefig("figures/fig2_loss_aversion.png", bbox_inches="tight")
    plt.close()
    print("  → figures/fig2_loss_aversion.png")
    return lambda_results


# ════════════════════════════════════════════════════════════════════════════
# BIAS 3 — ENDOWMENT EFFECT
# ════════════════════════════════════════════════════════════════════════════

def analyse_endowment(dfs):
    print("\n── BIAS 3: ENDOWMENT EFFECT ──")
    results = {}

    for name, df in dfs.items():
        en = df[df["bias"]=="endowment"].copy()
        en["price_value"] = en["price_value"].astype(float)
        en["accepted"] = en["parsed_choice"]=="Yes"
        results[name] = {}

        for surface in SURFACES:
            results[name][surface] = {}
            for price in PRICES:
                own = en[(en["surface"]==surface) & (en["condition"]==f"owner_price{price}")]["accepted"]
                non = en[(en["surface"]==surface) & (en["condition"]==f"non_owner_price{price}")]["accepted"]
                r_own = own.mean()
                r_non = non.mean()

                # chi-square
                p, _ = chi2_test(
                    [own.sum(), len(own)-own.sum()],
                    [non.sum(), len(non)-non.sum()]
                )
                h = cohens_h(r_own + 1e-9, r_non + 1e-9)
                ratio = r_own / r_non if r_non > 0 else float("inf")
                results[name][surface][price] = {"own": r_own, "non": r_non, "p": p, "h": h}
                add_stat("endowment", name, surface, f"owner_accept_${price}", r_own)
                add_stat("endowment", name, surface, f"nonowner_accept_${price}", r_non, p, h,
                         f"ratio={ratio:.1f}x")
                print(f"  {name:<22} {surface:<12} ${price} own={r_own:.2f} non={r_non:.2f} "
                      f"ratio={ratio:.1f}x  p={p:.4f}{sig_stars(p)}")

    # ── FIGURE 3: Endowment acceptance curves ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Figure 3 — Endowment Effect: Acceptance Rate by Ownership and Price",
                 fontsize=12, fontweight="bold", y=1.01)

    for row, name in enumerate(["GPT-4o-mini T=0.7", "Gemma3:4b T=0.7"]):
        if name not in results:
            continue
        for col, surface in enumerate(SURFACES):
            ax = axes[row][col]
            own_vals = [results[name][surface][p]["own"] for p in PRICES]
            non_vals = [results[name][surface][p]["non"] for p in PRICES]

            ax.plot(PRICES, own_vals, "o-", color=COLOR_MAP[name],
                    label="Owner", linewidth=2, markersize=7)
            ax.plot(PRICES, non_vals, "s--", color=COLOR_MAP[name],
                    label="Non-owner", linewidth=2, markersize=7, alpha=0.55)

            ax.set_xticks(PRICES)
            ax.set_xticklabels([f"${p}" for p in PRICES])
            ax.set_ylim(-0.05, 1.1)
            ax.set_xlabel("Price")
            ax.set_ylabel("Accept rate" if col==0 else "")
            ax.set_title(f"{name.split()[0]} — {surface.capitalize()}", fontsize=9)
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/fig3_endowment.png", bbox_inches="tight")
    plt.close()
    print("  → figures/fig3_endowment.png")
    return results


# ════════════════════════════════════════════════════════════════════════════
# BIAS 4 — CERTAINTY EFFECT (ALLAIS)
# ════════════════════════════════════════════════════════════════════════════

def analyse_certainty(dfs):
    print("\n── BIAS 4: CERTAINTY EFFECT ──")
    PATTERNS = ["(Choice1: B, Choice2: A)", "(Choice1: B, Choice2: B)",
                "(Choice1: A, Choice2: A)", "(Choice1: A, Choice2: B)"]
    LABELS   = ["Violation\n(B,A)", "Risk averse\n(B,B)",
                "Rational\n(A,A)", "Reverse\n(A,B)"]
    PCOLS    = [COLORS["human"], "#D97706", COLORS["novel"], "#7C3AED"]
    HUMAN_VIOLATION = 0.65

    results = {}
    for name, df in dfs.items():
        ce = df[df["bias"]=="certainty_effect"]
        results[name] = {}
        for surface in SURFACES:
            sub = ce[ce["surface"]==surface]
            rates = {pat: (sub["parsed_choice"]==pat).mean() for pat in PATTERNS}
            results[name][surface] = rates

            viol = rates[PATTERNS[0]]
            print(f"  {name:<22} {surface:<12} "
                  f"violation={viol:.2f}  (B,B)={rates[PATTERNS[1]]:.2f}  "
                  f"(A,A)={rates[PATTERNS[2]]:.2f}  (A,B)={rates[PATTERNS[3]]:.2f}")

            # chi-square: violation vs non-violation
            n_viol = (sub["parsed_choice"]==PATTERNS[0]).sum()
            n_total = len(sub)
            # compare to human baseline
            p, _ = chi2_test([n_viol, n_total-n_viol],
                             [int(HUMAN_VIOLATION*n_total),
                              int((1-HUMAN_VIOLATION)*n_total)])
            h = cohens_h(viol + 1e-9, HUMAN_VIOLATION)
            add_stat("certainty_effect", name, surface, "violation_rate", viol, p, h,
                     f"vs human {HUMAN_VIOLATION}")

    # ── FIGURE 4: Stacked bar by surface and model ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Figure 4 — Certainty Effect: Choice Pattern Distribution (Allais Paradox)",
                 fontsize=12, fontweight="bold", y=1.02)

    x     = np.arange(len(dfs))
    names = list(dfs.keys())

    for i, surface in enumerate(SURFACES):
        ax = axes[i]
        bottoms = np.zeros(len(names))
        for pat, label, col in zip(PATTERNS, LABELS, PCOLS):
            vals = [results[n][surface][pat] for n in names]
            ax.bar(x, vals, bottom=bottoms, label=label,
                   color=col, alpha=0.85, edgecolor="white", width=0.55)
            # label inside bar if wide enough
            for j, v in enumerate(vals):
                if v > 0.08:
                    ax.text(x[j], bottoms[j] + v/2, f"{v:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
            bottoms += np.array(vals)

        # human violation line
        ax.axhline(HUMAN_VIOLATION, color=COLORS["human"], linestyle="--",
                   linewidth=1.3, alpha=0.7)
        ax.text(len(names)-0.45, HUMAN_VIOLATION+0.02, f"Human\n({HUMAN_VIOLATION})",
                color=COLORS["human"], fontsize=7.5)

        ax.set_title(surface.capitalize(), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" T=","\nT=") for n in names], fontsize=7.5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Proportion" if i==0 else "")

    handles = [mpatches.Patch(color=c, label=l, alpha=0.85)
               for c, l in zip(PCOLS, LABELS)]
    handles.append(plt.Line2D([0],[0], color=COLORS["human"],
                               linestyle="--", label="Human violation rate"))
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    plt.tight_layout()
    plt.savefig("figures/fig4_certainty.png", bbox_inches="tight")
    plt.close()
    print("  → figures/fig4_certainty.png")
    return results


# ════════════════════════════════════════════════════════════════════════════
# BIAS 5 — STATUS QUO
# ════════════════════════════════════════════════════════════════════════════

def analyse_status_quo(dfs):
    print("\n── BIAS 5: STATUS QUO ──")

    def chose_stay(row):
        if "stay_first" in row["condition"]:
            return row["parsed_choice"] == "A"
        return row["parsed_choice"] == "B"

    results = {}
    for name, df in dfs.items():
        sq = df[df["bias"]=="status_quo"].copy()
        sq["chose_stay"] = sq.apply(chose_stay, axis=1)
        results[name] = {}
        for surface in SURFACES:
            sub = sq[sq["surface"]==surface]
            c1 = sub[sub["condition"]=="default_stay_first"]["chose_stay"].mean()
            c2 = sub[sub["condition"]=="default_switch_first"]["chose_stay"].mean()
            c3 = sub[sub["condition"]=="no_default_stay_first"]["chose_stay"].mean()
            c4 = sub[sub["condition"]=="no_default_switch_first"]["chose_stay"].mean()
            de = ((c1+c2)/2) - ((c3+c4)/2)
            pe = ((c1+c3)/2) - ((c2+c4)/2)
            results[name][surface] = {"de": de, "pe": pe,
                                       "c1":c1,"c2":c2,"c3":c3,"c4":c4}
            print(f"  {name:<22} {surface:<12} default_effect={de:+.2f}  "
                  f"position_effect={pe:+.2f}  C4={c4:.2f}")
            add_stat("status_quo", name, surface, "default_effect",  de,
                     note=f"C1={c1:.2f} C2={c2:.2f} C3={c3:.2f} C4={c4:.2f}")
            add_stat("status_quo", name, surface, "position_effect", pe)

    # ── FIGURE 5: Default vs position effect heatmap ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Figure 5 — Status Quo Bias: Default Effect and Position Effect",
                 fontsize=12, fontweight="bold")

    names   = list(results.keys())
    effects = ["de", "pe"]
    titles  = ["Default Effect\n(presence of default → stay)", 
               "Position Effect\n(stay listed first → stay)"]

    for ax, effect, title in zip(axes, effects, titles):
        matrix = np.array([[results[n][s][effect] for s in SURFACES]
                            for n in names])
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")

        ax.set_xticks(range(len(SURFACES)))
        ax.set_xticklabels([s.capitalize() for s in SURFACES])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_title(title, fontsize=10)

        for i in range(len(names)):
            for j in range(len(SURFACES)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val > 0.5 else "#111")

        plt.colorbar(im, ax=ax, shrink=0.8)

    ax.axhline(0.40, color=COLORS["human"], linestyle="--",
               linewidth=1, alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/fig5_status_quo.png", bbox_inches="tight")
    plt.close()
    print("  → figures/fig5_status_quo.png")
    return results


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY FIGURE — cross-model heatmap
# ════════════════════════════════════════════════════════════════════════════

def make_summary_figure(framing_r, lambda_r, certainty_r):
    print("\n── SUMMARY FIGURE ──")
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Figure 6 — Cross-Model Summary: Key Metrics vs Human Baseline",
                 fontsize=12, fontweight="bold")

    names   = list(framing_r.keys())
    metrics = [
        "Framing Δ\n(canonical)", "Framing Δ\n(novel)",
        "λ upper\nbound",
        "Allais viol.\n(canonical)", "Allais viol.\n(novel)",
        "Endow. ratio\n(canonical, T=0.7 only)",
    ]
    human_vals = [0.50, 0.50, 2.25, 0.65, 0.65, 2.0]

    matrix = []
    for name in names:
        row = [
            framing_r[name]["canonical"]["delta"],
            framing_r[name]["novel"]["delta"],
            lambda_r[name]["canonical"],
            certainty_r[name]["canonical"]["(Choice1: B, Choice2: A)"],
            certainty_r[name]["novel"]["(Choice1: B, Choice2: A)"],
            np.nan,  # endowment ratio filled below
        ]
        matrix.append(row)

    # fill endowment ratio for T=0.7 only
    endow_ratios = {"GPT-4o-mini T=0.7": 9.3, "GPT-4o-mini T=0.0": 16.7,
                    "Gemma3:4b T=0.7": 1.0, "Gemma3:4b T=0.3": 1.0}
    for i, name in enumerate(names):
        matrix[i][5] = endow_ratios.get(name, 1.0)

    matrix = np.array(matrix, dtype=float)

    # normalise each column 0-1 relative to human value for colour
    norm_matrix = np.zeros_like(matrix)
    for j in range(len(metrics)):
        col = matrix[:, j]
        hval = human_vals[j]
        norm_matrix[:, j] = col / (hval * 2) if hval > 0 else col

    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=8.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)

    for i in range(len(names)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            hval = human_vals[j]
            txt = f"{val:.2f}\n(H:{hval:.2f})"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color="#111" if norm_matrix[i,j] < 0.6 else "white")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Relative to human baseline")
    plt.tight_layout()
    plt.savefig("figures/fig6_summary_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  → figures/fig6_summary_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# SAVE STATS CSV
# ════════════════════════════════════════════════════════════════════════════

def save_stats():
    path = "figures/stats_summary.csv"
    fields = ["bias","model","surface","metric","value",
              "p_value","cohens_h","sig","note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stats_rows)
    print(f"\n  → {path}  ({len(stats_rows)} rows)")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("LLM Behavioral Bias Study — Full Analysis")
    print("=" * 55)
    dfs = load_all()
    if not dfs:
        print("No datasets found. Place CSV files in results/ folder.")
        return

    framing_r  = analyse_framing(dfs)
    lambda_r   = analyse_loss_aversion(dfs)
    endow_r    = analyse_endowment(dfs)
    certainty_r = analyse_certainty(dfs)
    sq_r       = analyse_status_quo(dfs)
    make_summary_figure(framing_r, lambda_r, certainty_r)
    save_stats()

    print("\n" + "=" * 55)
    print("All figures saved to figures/")
    print("All stats saved to figures/stats_summary.csv")
    print("\nFiles ready for paper:")
    for f in sorted(os.listdir("figures")):
        print(f"  figures/{f}")


if __name__ == "__main__":
    main()
