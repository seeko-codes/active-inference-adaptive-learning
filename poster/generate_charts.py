"""Generate all poster charts from simulation data."""

import sys
sys.path.insert(0, "/Users/aatutor/adaptive-learning")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

from simulation.monte_carlo import run_monte_carlo, run_population_monte_carlo
from simulation.analysis import compare_policies, learning_landscape, parameter_recovery
from simulation.learner_types import get_all_archetypes

# ── Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
})

POLICY_COLORS = {
    "active_inference": "#C17817",   # Gold accent (matches poster)
    "meta_function":    "#2E6B8A",   # Steel blue
    "fsrs_only":        "#7A7A7A",   # Gray
    "fixed_curriculum": "#A3A3A3",   # Light gray
    "random":           "#CCCCCC",   # Lightest gray
}

POLICY_LABELS = {
    "active_inference": "Active Inference",
    "meta_function":    "Meta-Function",
    "fsrs_only":        "FSRS-Only",
    "fixed_curriculum": "Fixed Curriculum",
    "random":           "Random",
}

POLICY_ORDER = ["active_inference", "meta_function", "fsrs_only", "fixed_curriculum", "random"]

OUT = "/Users/aatutor/adaptive-learning/poster"


# ═══════════════════════════════════════════
# RUN SIMULATION
# ═══════════════════════════════════════════
print("Running archetype Monte Carlo (8 types x 5 policies x 20 seeds)...")
# Use non-AI policies first (fast), then try AI
NON_AI = ["meta_function", "random", "fsrs_only", "fixed_curriculum"]
archetype_results = run_monte_carlo(
    policy_names=NON_AI,
    n_seeds=20,
    n_sessions=5,
    problems_per_session=20,
    base_seed=42,
)
print(f"  Non-AI: {len(archetype_results)} trajectories")

# Try active inference
try:
    ai_results = run_monte_carlo(
        policy_names=["active_inference"],
        n_seeds=20,
        n_sessions=5,
        problems_per_session=20,
        base_seed=42,
    )
    archetype_results.extend(ai_results)
    print(f"  AI: {len(ai_results)} trajectories")
    HAS_AI = True
except Exception as e:
    print(f"  AI policy failed ({e}), proceeding without it")
    HAS_AI = False

# Population run for aggregate stats
print("\nRunning population Monte Carlo (2000 students x non-AI policies)...")
pop_results = run_population_monte_carlo(
    n_students=2000,
    policy_names=NON_AI,
    n_sessions=5,
    problems_per_session=20,
    base_seed=42,
)
print(f"  Population: {len(pop_results)} trajectories")

# Analysis
comparisons = compare_policies(archetype_results)
landscape = learning_landscape(comparisons)
recovery = parameter_recovery(archetype_results)

# Aggregate by policy
policy_stats = defaultdict(list)
for c in comparisons:
    policy_stats[c.policy].append(c)

policy_means = {}
policy_stds = {}
for policy in POLICY_ORDER:
    if policy not in policy_stats and not HAS_AI and policy == "active_inference":
        continue
    comps = policy_stats.get(policy, [])
    if comps:
        masteries = [c.mean_final_mastery for c in comps]
        policy_means[policy] = np.mean(masteries)
        policy_stds[policy] = np.std(masteries)

# Population-level stats
pop_policy_stats = defaultdict(list)
for r in pop_results:
    pop_policy_stats[r.policy].append(r.final_mastery)

pop_means = {p: np.mean(v) for p, v in pop_policy_stats.items()}
pop_stds = {p: np.std(v) for p, v in pop_policy_stats.items()}

# ═══════════════════════════════════════════
# PRINT KEY NUMBERS (for poster placeholders)
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("POSTER PLACEHOLDER VALUES")
print("=" * 60)
print(f"N_STUDENTS: 2,000 (population) + 8 archetypes x 20 seeds")
print(f"N_SESSIONS: 5")
print(f"Total trajectories: {len(archetype_results) + len(pop_results):,}")
for p in POLICY_ORDER:
    if p in pop_means:
        print(f"  {POLICY_LABELS[p]:20s}: {pop_means[p]:.3f} +/- {pop_stds[p]:.3f}")
    elif p in policy_means:
        print(f"  {POLICY_LABELS[p]:20s}: {policy_means[p]:.3f} +/- {policy_stds[p]:.3f} (archetypes only)")

# Best/worst per landscape
print(f"\nLearning Landscape:")
for lt, best_p in landscape["best_policy"].items():
    adv = landscape["advantage"].get(lt, 0)
    vuln = landscape["vulnerability"].get(lt, 0)
    print(f"  {lt:22s}: best={best_p}, advantage={adv:.4f}, vulnerability={vuln:.3f}")

# Most vulnerable
if landscape["vulnerability"]:
    most_vuln = max(landscape["vulnerability"], key=landscape["vulnerability"].get)
    print(f"\nMost vulnerable to policy choice: {most_vuln} (vulnerability={landscape['vulnerability'][most_vuln]:.3f})")

# Best advantage
if landscape["advantage"]:
    best_adv_lt = max(landscape["advantage"], key=landscape["advantage"].get)
    print(f"Largest advantage: {best_adv_lt} — {landscape['best_policy'][best_adv_lt]} by +{landscape['advantage'][best_adv_lt]:.3f}")

print(f"\nParameter Recovery:")
print(f"  Schema accuracy: {recovery['schema_accuracy']:.1%} (n={recovery['schema_n']})")
print(f"  RS MAE: {recovery['rs_mae']:.3f} (n={recovery['rs_n']})")


# ═══════════════════════════════════════════
# CHART 1: Policy Comparison (bar chart)
# ═══════════════════════════════════════════
print("\nGenerating Chart 1: Policy Comparison...")

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

policies_present = [p for p in POLICY_ORDER if p in pop_means or p in policy_means]
means = [pop_means.get(p, policy_means.get(p, 0)) for p in policies_present]
stds = [pop_stds.get(p, policy_stds.get(p, 0)) for p in policies_present]
colors = [POLICY_COLORS[p] for p in policies_present]
labels = [POLICY_LABELS[p] for p in policies_present]

bars = ax.bar(range(len(policies_present)), means, yerr=stds, capsize=5,
              color=colors, edgecolor="#222222", linewidth=0.8, width=0.65,
              error_kw={"linewidth": 1.2, "color": "#333333"})

# Value labels on bars
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.008,
            f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(range(len(policies_present)))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Mean Final Mastery", fontsize=11)
ax.set_title("Policy Comparison", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, max(means) * 1.18)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

plt.tight_layout()
fig.savefig(f"{OUT}/chart1_policy_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: chart1_policy_comparison.png")


# ═══════════════════════════════════════════
# CHART 2: Learning Trajectories (line chart)
# ═══════════════════════════════════════════
print("Generating Chart 2: Learning Trajectories...")

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

# Get mean trajectories per policy from population results
pop_traj_by_policy = defaultdict(list)
for r in pop_results:
    pop_traj_by_policy[r.policy].append(r.mastery_trajectory)

# Also use archetype results for AI if available
for r in archetype_results:
    if r.policy == "active_inference":
        pop_traj_by_policy[r.policy].append(r.mastery_trajectory)

for policy in POLICY_ORDER:
    trajs = pop_traj_by_policy.get(policy, [])
    if not trajs:
        continue
    max_len = max(len(t) for t in trajs)
    padded = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
    mean_traj = np.mean(padded, axis=0)
    std_traj = np.std(padded, axis=0)

    # x-axis: session points (initial, post-session, post-forgetting, ...)
    x = np.arange(len(mean_traj))

    ax.plot(x, mean_traj, color=POLICY_COLORS[policy], linewidth=2.2,
            label=POLICY_LABELS[policy], zorder=3)
    ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj,
                     color=POLICY_COLORS[policy], alpha=0.12, zorder=1)

# Mark session boundaries
# Pattern: [initial, post-S1, post-forget, post-S2, post-forget, ...]
# Sessions at indices 0, 1, 3, 5, 7, 9 (every other after first)
session_labels_x = [1, 3, 5, 7, 9]
for i, sx in enumerate(session_labels_x):
    if sx < len(mean_traj):
        ax.axvline(sx, color="#DDDDDD", linewidth=0.8, linestyle="--", zorder=0)

# Mark forgetting intervals
forget_x = [2, 4, 6, 8]
for fx in forget_x:
    if fx < len(mean_traj):
        ax.axvspan(fx - 0.3, fx + 0.3, color="#FFF3E0", alpha=0.4, zorder=0)

ax.set_xlabel("Simulation Step  (sessions + forgetting intervals)", fontsize=10)
ax.set_ylabel("Mean Mastery", fontsize=11)
ax.set_title("Learning Trajectories Across Sessions", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC")
ax.set_ylim(0, None)

plt.tight_layout()
fig.savefig(f"{OUT}/chart2_learning_trajectories.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: chart2_learning_trajectories.png")


# ═══════════════════════════════════════════
# CHART 3: Learning Landscape (heatmap)
# ═══════════════════════════════════════════
print("Generating Chart 3: Learning Landscape...")

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

# Build matrix: rows = learner types, cols = policies
learner_types = sorted(set(c.learner_type for c in comparisons))
policies_in_data = sorted(set(c.policy for c in comparisons),
                          key=lambda p: POLICY_ORDER.index(p) if p in POLICY_ORDER else 99)

matrix = np.zeros((len(learner_types), len(policies_in_data)))
for c in comparisons:
    i = learner_types.index(c.learner_type)
    j = policies_in_data.index(c.policy)
    matrix[i, j] = c.mean_final_mastery

# Heatmap
im = ax.imshow(matrix, cmap="YlOrBr", aspect="auto", vmin=matrix.min() * 0.95)

# Annotate cells
for i in range(len(learner_types)):
    row_max = matrix[i, :].max()
    for j in range(len(policies_in_data)):
        val = matrix[i, j]
        is_best = abs(val - row_max) < 1e-6
        weight = "bold" if is_best else "normal"
        color = "white" if val > (matrix.max() + matrix.min()) / 2 else "#222222"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=8.5, fontweight=weight, color=color)

ax.set_xticks(range(len(policies_in_data)))
ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in policies_in_data],
                    fontsize=9, rotation=25, ha="right")
ax.set_yticks(range(len(learner_types)))
lt_display = [lt.replace("_", " ").title() for lt in learner_types]
ax.set_yticklabels(lt_display, fontsize=9)

ax.set_title("Learning Landscape: Policy x Learner Type → Mastery",
             fontsize=13, fontweight="bold", pad=12)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Final Mastery", fontsize=10)

plt.tight_layout()
fig.savefig(f"{OUT}/chart3_learning_landscape.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: chart3_learning_landscape.png")


# ═══════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL CHARTS GENERATED")
print("=" * 60)
print(f"Files in {OUT}/:")
print("  chart1_policy_comparison.png")
print("  chart2_learning_trajectories.png")
print("  chart3_learning_landscape.png")
