#!/usr/bin/env python3
"""Generate publication figures for M7 (NeurIPS 2026) from measured data only.

RULE 1: never put a number in the paper that is not in a CSV. No estimates.

This script used to violate that rule in three places:
  * Figure 3 (gate trajectory) was drawn from HARDCODED literal arrays.
  * Figure 4 (memory footprint) was drawn from a HARDCODED [80, 150, 4] MB.
  * Figure 1 drew "single-task" reference lines at y=97.68 / y=82.16 -- hardcoded numbers
    lifted from the LEGACY BI-MODAL paper (M3 and M2 respectively), mislabelled and
    overlaid on a tri-modal plot.
All three were titled or captioned as if they were 3-seed results from this experiment.

Every figure here is now either (a) plotted from a CSV, or (b) derived from CONFIG with the
arithmetic printed, or (c) NOT DRAWN AT ALL. A figure that cannot be sourced is skipped
loudly rather than fabricated quietly.
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid')
RESULTS_DIR = Path('./ICONS_M7/results')
PLOT_DIR = RESULTS_DIR / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'nmnist': '#2E86AB', 'shd': '#A23B72', 'dvs': '#F18F01',
    'noreplay': '#6B7280', 'replay': '#10B981',
    'hopfield': '#3B82F6', 'hgrn': '#EF4444',
}

CONDITIONS = ['no_replay', 'replay']


# ═════════════════════════════════════════════════════════════════════════════
#  LOAD  (mirrors .claude/skills/analyze-cl-results/scripts/load_results.py;
#          kept inline so this script stays standalone for RunPod / CI)
# ═════════════════════════════════════════════════════════════════════════════
def load_summary(csv_path):
    """Return one row per (seed, condition) -- the only valid analysis population.

    The CSV interleaves TWO schemas: incremental rows carry a `step`, summary rows do not.
    Re-runs append rather than overwrite.

    The documented ritual is df[df['step'].isna()].drop_duplicates() -- but drop_duplicates()
    is EXACT-match, so a re-run whose summary row differs in the last float digit leaves BOTH
    rows behind. Aggregating over rows would then inflate n and compute the +/- std across a
    duplicated seed instead of independent ones: a silently wrong error bar. So we collapse to
    one row per (seed, condition) explicitly.
    """
    if not csv_path.exists():
        sys.exit(
            f"\nNO RESULTS. {csv_path} does not exist.\n"
            f"Nothing can be plotted. Run the experiment first (see the /run-m7-experiment\n"
            f"skill). This is the expected state before any training run has completed.\n"
        )

    df = pd.read_csv(csv_path)
    summary = df[df['step'].isna() | (df['step'].astype(str).str.strip() == '')].copy()
    summary = summary.drop_duplicates()

    dupes = summary.groupby(['seed', 'condition']).size()
    dupes = dupes[dupes > 1]
    if len(dupes):
        print("⚠️  NEAR-DUPLICATE SUMMARY ROWS (same seed+condition, differing values):")
        for (seed, cond), n in dupes.items():
            print(f"      seed={seed} condition={cond} appears {n}x")
        print("    Keeping the FIRST row per (seed, condition). The CSV needs cleaning, and")
        print("    any previously published ± std drawn from it should be re-checked.\n")

    return summary.groupby(['seed', 'condition'], as_index=False).first()


def aggregate(summary, cond):
    """mean/std across INDEPENDENT SEEDS for one condition, or None if absent."""
    sub = summary[summary['condition'] == cond]
    if sub.empty:
        return None
    return {
        'n_seeds': sub['seed'].nunique(),
        'seeds': sorted(sub['seed'].unique().tolist()),
        'mean': sub.mean(numeric_only=True),
        'std': sub.std(numeric_only=True, ddof=1).fillna(0.0),
    }


csv_path = RESULTS_DIR / 'cl_trimodal_results.csv'
summary = load_summary(csv_path)

stats = {c: aggregate(summary, c) for c in CONDITIONS}
missing = [c for c, s in stats.items() if s is None]

if missing:
    # The old script indexed .values[0] here and died with a bare IndexError.
    sys.exit(
        f"\nINCOMPLETE RESULTS: condition(s) {missing} are absent from {csv_path}.\n"
        f"Present: {sorted(summary['condition'].unique().tolist())}\n"
        f"Figures 1-2 compare no_replay vs replay and cannot be drawn without both.\n"
        f"Run the missing condition, or use /analyze-cl-results to inspect what you have.\n"
    )

# The seed count comes from the DATA. It is never asserted in a title.
seed_sets = {c: stats[c]['seeds'] for c in CONDITIONS}
n_seeds = min(stats[c]['n_seeds'] for c in CONDITIONS)
seed_label = f"{n_seeds} seed{'s' if n_seeds != 1 else ''}"
if n_seeds < 2:
    seed_label += " — NO ERROR BARS POSSIBLE"

print(f"Loaded {csv_path}")
for c in CONDITIONS:
    print(f"  {c:<10} seeds={seed_sets[c]}  (n={stats[c]['n_seeds']})")
if seed_sets['no_replay'] != seed_sets['replay']:
    print("⚠️  The two conditions were run on DIFFERENT seed sets. They are not paired.")
expected = {42, 123, 456}
for c in CONDITIONS:
    gap = expected - set(seed_sets[c])
    if gap:
        print(f"⚠️  {c}: missing documented seed(s) {sorted(gap)} — this is NOT a 3-seed result.")
print()


def m(cond, col):
    return float(stats[cond]['mean'][col])


def s(cond, col):
    return float(stats[cond]['std'][col])


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Accuracy by condition
# ═════════════════════════════════════════════════════════════════════════════
tasks = ['T1 N-MNIST', 'T2 SHD', 'T3 DVS']
acc_cols = ['T1_after_T1', 'T2_after_T2', 'T3_after_T3']

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(tasks))
width = 0.28

means_nr = [m('no_replay', c) for c in acc_cols]
stds_nr = [s('no_replay', c) for c in acc_cols]
means_r = [m('replay', c) for c in acc_cols]
stds_r = [s('replay', c) for c in acc_cols]

bars1 = ax.bar(x - width / 2, means_nr, width, yerr=stds_nr, label='No Replay',
               color=COLORS['noreplay'], edgecolor='black', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})
bars2 = ax.bar(x + width / 2, means_r, width, yerr=stds_r, label='+ Replay',
               color=COLORS['replay'], edgecolor='black', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})

for bars, means, stds in ((bars1, means_nr, stds_nr), (bars2, means_r, stds_r)):
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f'{mean:.1f}±{std:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + std),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

# NOTE: the previous version drew reference lines at y=97.68 and y=82.16 labelled
# "N-MNIST single-task" / "SHD single-task". Those were hardcoded numbers from the LEGACY
# BI-MODAL paper (M3's 97.68 and M2's 82.16) -- a different experiment, mislabelled and
# overlaid here. They are not in this CSV, so they are not drawn. If single-task baselines
# are wanted, run them and add them to the CSV.

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title(f'M7 Tri-Modal CL: Accuracy by Condition ({seed_label}, mean ± std)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(65, 105)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'fig1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 1: accuracy — from CSV")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Forgetting
# ═════════════════════════════════════════════════════════════════════════════
forget_cols = ['forgetting_T1_T2', 'forgetting_T1_T3', 'forgetting_T2_T3']
labels = ['T1→T2', 'T1→T3', 'T2→T3']

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(labels))

fm_nr = [m('no_replay', c) for c in forget_cols]
fs_nr = [s('no_replay', c) for c in forget_cols]
fm_r = [m('replay', c) for c in forget_cols]
fs_r = [s('replay', c) for c in forget_cols]

bars1 = ax.bar(x - width / 2, fm_nr, width, yerr=fs_nr, label='No Replay',
               color=COLORS['noreplay'], edgecolor='black', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})
bars2 = ax.bar(x + width / 2, fm_r, width, yerr=fs_r, label='+ Replay',
               color=COLORS['replay'], edgecolor='black', linewidth=0.5,
               capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})

for bars, means, stds in ((bars1, fm_nr, fs_nr), (bars2, fm_r, fs_r)):
    for bar, mean, std in zip(bars, means, stds):
        h = bar.get_height()
        ax.annotate(f'{mean:+.2f}', xy=(bar.get_x() + bar.get_width() / 2,
                                        h + (std if h >= 0 else -std)),
                    xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                    ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=8, fontweight='bold')

ax.set_ylabel('Forgetting (pp)', fontsize=12)
# RULE 5: the claim is ~0 pp forgetting, NOT "eliminated" / "100% reduction".
ax.set_title(f'M7: Forgetting Across Task Transitions ({seed_label}, mean ± std)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'fig2_forgetting.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 2: forgetting — from CSV")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Adaptive gate trajectory  (MEASURED, or not drawn)
# ═════════════════════════════════════════════════════════════════════════════
# This is the paper's core novelty claim: the gate only learns to use HGRN under replay.
# It used to be drawn from hardcoded literals. m7_trimodal_run.py already MEASURES the
# per-epoch gate split -- it just never wrote it to disk. It does now
# (_save_gate_history -> gate_trajectory.csv). If that file is absent, we skip the figure.
gate_path = RESULTS_DIR / 'gate_trajectory.csv'

if not gate_path.exists():
    print("⏭️  Figure 3 SKIPPED — no gate_trajectory.csv.")
    print("    The gate trajectory is the paper's central claim and MUST NOT be drawn from")
    print("    hardcoded values. Re-run m7_trimodal_run.py (it now persists the measurement)")
    print("    and this figure will render from real data.")
else:
    gate = pd.read_csv(gate_path)
    gate_t2 = gate[gate['task'] == 'T2_shd']  # gate specialisation is claimed during SHD
    if gate_t2.empty:
        print("⏭️  Figure 3 SKIPPED — gate_trajectory.csv has no T2_shd rows.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        drew = False
        for cond, style, alpha in (('no_replay', '-', 1.0), ('replay', '--', 0.6)):
            sub = gate_t2[gate_t2['condition'] == cond]
            if sub.empty:
                print(f"    (no {cond} rows in gate_trajectory.csv — omitted from Fig 3)")
                continue
            g = sub.groupby('epoch').agg(
                hop_m=('gate_hop', 'mean'), hop_s=('gate_hop', 'std'),
                hgrn_m=('gate_hgrn', 'mean'), hgrn_s=('gate_hgrn', 'std'),
            ).fillna(0.0).reset_index()
            ax.errorbar(g['epoch'], g['hop_m'], yerr=g['hop_s'], fmt='o' + style,
                        color=COLORS['hopfield'], linewidth=2.5, markersize=7, alpha=alpha,
                        capsize=3, label=f'Hopfield ({cond})')
            ax.errorbar(g['epoch'], g['hgrn_m'], yerr=g['hgrn_s'], fmt='s' + style,
                        color=COLORS['hgrn'], linewidth=2.5, markersize=7, alpha=alpha,
                        capsize=3, label=f'HGRN ({cond})')
            drew = True

        if drew:
            n_gate_seeds = gate_t2['seed'].nunique()
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Gate Weight', fontsize=12)
            ax.set_title(
                f'Adaptive Gate During SHD Training '
                f'({n_gate_seeds} seed{"s" if n_gate_seeds != 1 else ""}, mean ± std)',
                fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='center right')
            ax.set_ylim(-0.02, 1.05)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / 'fig3_gate_trajectory.png', dpi=300, bbox_inches='tight')
            print(f"✅ Figure 3: gate trajectory — from CSV ({n_gate_seeds} seeds)")
        plt.close()

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Replay memory footprint  (DERIVED from CONFIG, arithmetic shown)
# ═════════════════════════════════════════════════════════════════════════════
# Was hardcoded as [80, 150, 4] MB with no source. These are now derived from the actual
# buffer and input configuration, and the arithmetic is printed so it can be audited.
SAMPLES_PER_CLASS = 50      # EmbeddingReplayBuffer(samples_per_class=50)
FEATURE_DIM = 512           # EmbeddingReplayBuffer(feature_dim=512)
BYTES = 4                   # float32
CLASSES = {'nmnist': 10, 'shd': 20, 'dvs': 11}

# Dense input frames -- what the model actually consumes (see the collate fns).
FRAME_ELEMS = {
    'nmnist': 25 * 2 * 34 * 34,      # [T=25, C=2, 34, 34]
    'shd': 100 * 1 * 1 * 700,        # [T=100, 700 channels]
    'dvs': 25 * 2 * 34 * 34,         # pooled 128->34 to match the visual backbone
}

n_classes = sum(CLASSES.values())
emb_bytes = n_classes * SAMPLES_PER_CLASS * FEATURE_DIM * BYTES
raw_bytes = sum(CLASSES[k] * SAMPLES_PER_CLASS * FRAME_ELEMS[k] * BYTES for k in CLASSES)

MB = 1024 ** 2
emb_mb, raw_mb = emb_bytes / MB, raw_bytes / MB

print("\n  Figure 4 derivation (audit this):")
print(f"    classes                = {n_classes}  ({CLASSES})")
print(f"    embeddings  {n_classes} x {SAMPLES_PER_CLASS} x {FEATURE_DIM} x {BYTES}B "
      f"= {emb_bytes:,} B = {emb_mb:.2f} MiB")
print(f"    raw frames  sum(classes x {SAMPLES_PER_CLASS} x frame_elems x {BYTES}B) "
      f"= {raw_bytes:,} B = {raw_mb:.1f} MiB")
print(f"    compression = {raw_bytes / emb_bytes:.1f}x\n")

fig, ax = plt.subplots(figsize=(7, 4))
methods = ['Raw dense frames\n(50/class)', 'Embedding replay\n(ours)']
vals = [raw_mb, emb_mb]
bars = ax.barh(methods, vals, color=['#6B7280', COLORS['replay']],
               edgecolor='black', linewidth=0.5, height=0.5)
for bar, v in zip(bars, vals):
    ax.annotate(f'{v:.1f} MiB', xy=(v, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center',
                fontsize=10, fontweight='bold')
ax.set_xlabel('Replay memory for all 3 tasks (MiB)', fontsize=12)
ax.set_title(f'Replay Buffer Footprint — {raw_bytes / emb_bytes:.0f}× compression\n'
             f'(derived from CONFIG, not measured)', fontsize=12, fontweight='bold')
ax.set_xlim(0, raw_mb * 1.25)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'fig4_memory_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 4: memory — derived from CONFIG (arithmetic above)")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Architecture schematic  (illustrative; no data claims)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')


def box(ax, x, y, w, h, text, color, fontsize=9):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                         facecolor=color, edgecolor='black', linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', wrap=True)


box(ax, 0.2, 4.8, 1.8, 0.8, 'N-MNIST\nVisual', COLORS['nmnist'], 8)
box(ax, 0.2, 3.0, 1.8, 0.8, 'SHD\nAudio', COLORS['shd'], 8)
box(ax, 0.2, 1.2, 1.8, 0.8, 'DVS-Gesture\nMotion', COLORS['dvs'], 8)
box(ax, 2.5, 4.8, 1.5, 0.8, 'Conv2D\nBackbone', '#E5E7EB', 8)
box(ax, 2.5, 3.0, 1.5, 0.8, 'Linear\nBackbone', '#E5E7EB', 8)
box(ax, 2.5, 1.2, 1.5, 0.8, 'Conv2D\nBackbone', '#E5E7EB', 8)
box(ax, 4.5, 2.5, 1.2, 1.5, '512-d\nFeature\nSpace', '#FEF3C7', 9)
box(ax, 6.0, 4.0, 1.4, 0.9, 'Hopfield\nMemory', COLORS['hopfield'], 8)
box(ax, 6.0, 1.6, 1.4, 0.9, 'HGRN\nGate', COLORS['hgrn'], 8)
box(ax, 6.0, 2.9, 1.4, 0.7, 'Adaptive\nGate', '#D1FAE5', 8)
box(ax, 7.8, 2.5, 1.2, 1.5, 'Weighted\nCombination', '#E0E7FF', 9)
box(ax, 9.3, 4.0, 0.6, 0.6, 'T1', COLORS['nmnist'], 9)
box(ax, 9.3, 2.7, 0.6, 0.6, 'T2', COLORS['shd'], 9)
box(ax, 9.3, 1.4, 0.6, 0.6, 'T3', COLORS['dvs'], 9)

for y in (5.2, 3.4, 1.6):
    ax.annotate('', xy=(2.5, y), xytext=(2.0, y), arrowprops=dict(arrowstyle='->', color='black'))
for xy, xytext in (((4.5, 3.8), (4.0, 5.2)), ((4.5, 3.3), (4.0, 3.4)), ((4.5, 2.8), (4.0, 1.6))):
    ax.annotate('', xy=xy, xytext=xytext, arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.annotate('', xy=(6.0, 4.5), xytext=(5.7, 3.5), arrowprops=dict(arrowstyle='->', color=COLORS['hopfield'], lw=1.5))
ax.annotate('', xy=(6.0, 2.5), xytext=(5.7, 3.0), arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.annotate('', xy=(6.0, 2.0), xytext=(5.7, 2.8), arrowprops=dict(arrowstyle='->', color=COLORS['hgrn'], lw=1.5))
ax.annotate('', xy=(7.8, 3.25), xytext=(7.4, 4.45), arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('', xy=(7.8, 3.25), xytext=(7.4, 2.05), arrowprops=dict(arrowstyle='->', color='black'))
for xy, xytext in (((9.3, 4.3), (9.0, 3.5)), ((9.3, 3.0), (9.0, 3.25)), ((9.3, 1.7), (9.0, 2.5))):
    ax.annotate('', xy=xy, xytext=xytext, arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ax.set_title('M7 Architecture: Modality-Adaptive SNN with Embedding Replay (schematic)',
             fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'fig5_architecture_schematic.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 5: architecture schematic (illustrative — asserts no numbers)")

print(f"\n📊 Figures written to {PLOT_DIR}")
print("   Reminder: report TEST accuracy (SHD val/test gap is 15-19pp), and pair every")
print("   accuracy with its spike rate / sparsity and timestep count.")
