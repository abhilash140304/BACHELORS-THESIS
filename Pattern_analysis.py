"""
=============================================================================
PATTERN ANALYSIS: Real vs Synthetic Session Behavior
=============================================================================
Compares real user sessions with first-order and second-order Markov-generated
synthetic sessions across four dimensions:
  1. Action distribution
  2. Transition distribution (Jensen-Shannon divergence)
  3. Session length distribution
  4. N-gram comparison (bigrams / trigrams)

Also extracts:
  - Transition hotspots (top-K transitions)
  - Behavioral insights (loops, entry/exit, dominant paths)

Outputs:
  - Printed summaries
  - Matplotlib plots (optional)
  - results.json
=============================================================================
"""

import json
import math
import collections
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Optional: set a nicer style ──────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass  # fallback to default

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

# File paths  ── adjust if your files live elsewhere
TRAIN_PATH      = Path("train_sequences.json")
TEST_PATH       = Path("test_sequences.json")
GENERATED_PATH  = Path("synthetic_sessions.json")

TOP_K           = 20          # top-K transitions to report
NGRAM_TOP_K     = 15          # top-K n-grams to report
PLOT            = True        # set False to skip all matplotlib output
OUTPUT_JSON     = Path("results.json")

REAL_COLOR      = "#2E86AB"
FIRST_COLOR     = "#E84855"
SECOND_COLOR    = "#3BB273"


# =============================================================================
# 1.  DATA LOADING
# =============================================================================

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_all_data():
    """Load every dataset and return a unified dict."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train = load_json(TRAIN_PATH)
    test  = load_json(TEST_PATH)
    gen   = load_json(GENERATED_PATH)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _extract_real(obj):
        """
        Pull encoded integer sequences from a real-session JSON file.
        Prefers 'encoded_sequences' (guaranteed int indices, matching the
        synthetic format) over 'sequences' (human-readable string tokens).
        Falls back gracefully if neither key is found.
        """
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "encoded_sequences" in obj:
                return obj["encoded_sequences"]
            for key in ("sequences", "sessions", "data"):
                if key in obj:
                    return obj[key]
        return list(obj.values())[0]

    def _unwrap_syn(obj):
        """Unwrap synthetic-session lists (already integers)."""
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("sessions", "data", "sequences"):
                if key in obj:
                    return obj[key]
        return list(obj.values())[0]

    def _to_int_seq(sessions):
        """
        Coerce every token to int.
        Real sessions stored as string action names ('home_view') cannot be
        cast to int — those are returned as uniform str lists instead.
        This prevents the mixed int/str TypeError during sorted().
        """
        result = []
        for seq in sessions:
            converted = []
            all_int = True
            for a in seq:
                try:
                    converted.append(int(a))
                except (ValueError, TypeError):
                    all_int = False
                    break
            if all_int:
                result.append(converted)
            else:
                result.append([str(a) for a in seq])
        return result

    def _build_index_to_action(obj):
        """
        Reconstruct an index->action label mapping from the real-data JSON
        by zipping parallel 'sequences' (strings) and 'encoded_sequences' (ints).
        """
        mapping = {}
        if not isinstance(obj, dict):
            return mapping
        raw = obj.get("sequences", [])
        enc = obj.get("encoded_sequences", [])
        if raw and enc and len(raw) == len(enc):
            for str_seq, int_seq in zip(raw, enc):
                for lbl, idx in zip(str_seq, int_seq):
                    mapping[str(idx)] = str(lbl)
        return mapping

    # ── load real sessions (use encoded integer sequences) ────────────────────
    real_sessions = _to_int_seq(_extract_real(train) + _extract_real(test))

    # ── load synthetic sessions (already integers) ────────────────────────────
    first_sessions  = _to_int_seq(_unwrap_syn(gen.get("first_order_sessions",
                                                       gen.get("first_order", []))))
    second_sessions = _to_int_seq(_unwrap_syn(gen.get("second_order_sessions",
                                                       gen.get("second_order", []))))

    # ── build index->action label mapping ─────────────────────────────────────
    action_to_index = gen.get("action_to_index", {})
    index_to_action = gen.get("index_to_action", {})
    if not index_to_action and action_to_index:
        index_to_action = {str(v): k for k, v in action_to_index.items()}
    if not index_to_action:
        index_to_action = _build_index_to_action(train)
    if not index_to_action:
        index_to_action = _build_index_to_action(test)

    if index_to_action:
        print(f"  Label mapping   : {len(index_to_action)} actions resolved")

    print(f"  Real sessions   : {len(real_sessions)}")
    print(f"  First-order syn : {len(first_sessions)}")
    print(f"  Second-order syn: {len(second_sessions)}")

    # Unique actions — safe to sort because all tokens are now the same type
    all_actions = sorted({a for seq in real_sessions + first_sessions + second_sessions
                           for a in seq})
    print(f"  Unique actions  : {len(all_actions)}")
    print()

    return {
        "real":   real_sessions,
        "first":  first_sessions,
        "second": second_sessions,
        "all_actions":    all_actions,
        "index_to_action": index_to_action,
    }


def label(idx, index_to_action):
    """Return human-readable label for an action index."""
    key = str(idx)
    return index_to_action.get(key, str(idx))


# =============================================================================
# 2.  ACTION DISTRIBUTION
# =============================================================================

def compute_action_distribution(sessions: list, all_actions: list) -> dict:
    """
    Count action frequencies across all sessions,
    return normalised probability for every known action.
    """
    counter = collections.Counter(a for seq in sessions for a in seq)
    total   = max(sum(counter.values()), 1)
    return {a: counter.get(a, 0) / total for a in all_actions}


def print_action_distribution_summary(dists: dict, top_n: int = 10):
    print("─" * 60)
    print("ACTION DISTRIBUTION  (top-{} actions by real frequency)".format(top_n))
    print("─" * 60)
    top_actions = sorted(dists["real"], key=lambda a: dists["real"][a], reverse=True)[:top_n]
    header = f"{'Action':>10}  {'Real':>8}  {'1st-ord':>8}  {'2nd-ord':>8}"
    print(header)
    print("-" * len(header))
    for a in top_actions:
        print(f"{str(a):>10}  {dists['real'][a]:>8.4f}  "
              f"{dists['first'][a]:>8.4f}  {dists['second'][a]:>8.4f}")
    print()


def plot_action_distributions(dists: dict, all_actions: list, index_to_action: dict):
    """Bar chart comparing action distributions (top-30 by real frequency)."""
    top_actions = sorted(all_actions,
                         key=lambda a: dists["real"][a], reverse=True)[:30]
    labels = [label(a, index_to_action) for a in top_actions]
    x = np.arange(len(top_actions))
    width = 0.27

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width, [dists["real"][a]   for a in top_actions], width, label="Real",         color=REAL_COLOR)
    ax.bar(x,          [dists["first"][a]  for a in top_actions], width, label="1st-order syn", color=FIRST_COLOR)
    ax.bar(x + width,  [dists["second"][a] for a in top_actions], width, label="2nd-order syn", color=SECOND_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Normalised frequency")
    ax.set_title("Action Distribution: Real vs Synthetic (top-30 actions)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("action_distribution.png", dpi=120)
    plt.show()
    print("  → Saved action_distribution.png\n")


# =============================================================================
# 3.  TRANSITION MATRIX & JENSEN-SHANNON DIVERGENCE
# =============================================================================

def compute_transition_matrix(sessions: list, all_actions: list) -> np.ndarray:
    """
    Build a row-normalised bigram transition matrix.
    Shape: (|A|, |A|)  where  M[i,j] = P(j | i)
    """
    n = len(all_actions)
    idx = {a: i for i, a in enumerate(all_actions)}
    counts = np.zeros((n, n), dtype=np.float64)

    for seq in sessions:
        for a, b in zip(seq[:-1], seq[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1

    # Row-normalise (add tiny epsilon to avoid divide-by-zero)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def _js_div_1d(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two 1-D probability vectors."""
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def compute_js_divergence(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
    """
    Average row-wise JSD between two transition matrices.
    Only averages over rows that have at least one non-zero entry in mat_a.
    Returns a value in [0, ln(2)] (nats).
    """
    active_rows = mat_a.sum(axis=1) > 0
    if not active_rows.any():
        return float("nan")
    divs = [_js_div_1d(mat_a[i].copy(), mat_b[i].copy())
            for i in range(mat_a.shape[0]) if active_rows[i]]
    return float(np.mean(divs))


def print_transition_divergence(real_mat, first_mat, second_mat):
    jsd_first  = compute_js_divergence(real_mat, first_mat)
    jsd_second = compute_js_divergence(real_mat, second_mat)
    print("─" * 60)
    print("TRANSITION DIVERGENCE  (Jensen-Shannon, row-averaged)")
    print("─" * 60)
    print(f"  Real vs 1st-order : JSD = {jsd_first:.6f} nats")
    print(f"  Real vs 2nd-order : JSD = {jsd_second:.6f} nats")
    winner = "2nd-order" if jsd_second < jsd_first else "1st-order"
    print(f"  → {winner} synthetic sessions are closer to real transitions")
    print()
    return {"real_vs_first": jsd_first, "real_vs_second": jsd_second}


def plot_transition_heatmaps(real_mat, first_mat, second_mat,
                             all_actions, index_to_action, top_n=20):
    """Heatmaps of the transition matrices for the top-N most active actions."""
    # Clamp top_n to the actual number of actions so ticks always match labels
    top_n = min(top_n, len(all_actions))

    # Select top-N actions by row-activity in real matrix
    active  = real_mat.sum(axis=1)
    top_idx = np.argsort(active)[::-1][:top_n]
    actual_n = len(top_idx)          # may be < top_n if vocab is small

    sub_real   = real_mat[np.ix_(top_idx, top_idx)]
    sub_first  = first_mat[np.ix_(top_idx, top_idx)]
    sub_second = second_mat[np.ix_(top_idx, top_idx)]
    tick_labels = [label(all_actions[i], index_to_action) for i in top_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, mat, title in zip(axes,
                               [sub_real, sub_first, sub_second],
                               ["Real", "1st-order Synthetic", "2nd-order Synthetic"]):
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=max(sub_real.max(), 1e-9))
        ax.set_title(title)
        # Use actual_n — never hardcode top_n here
        ax.set_xticks(range(actual_n))
        ax.set_xticklabels(tick_labels, rotation=70, ha="right", fontsize=6)
        ax.set_yticks(range(actual_n))
        ax.set_yticklabels(tick_labels, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"Transition Matrices (top-{actual_n} actions)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("transition_heatmaps.png", dpi=120)
    plt.show()
    print("  → Saved transition_heatmaps.png\n")


# =============================================================================
# 4.  SESSION LENGTH DISTRIBUTION
# =============================================================================

def compute_length_stats(sessions: list) -> dict:
    lengths = [len(s) for s in sessions]
    return {
        "mean": float(np.mean(lengths)),
        "std":  float(np.std(lengths)),
        "min":  int(np.min(lengths)),
        "max":  int(np.max(lengths)),
        "p25":  float(np.percentile(lengths, 25)),
        "p50":  float(np.percentile(lengths, 50)),
        "p75":  float(np.percentile(lengths, 75)),
    }


def print_length_stats(stats: dict):
    print("─" * 60)
    print("SESSION LENGTH DISTRIBUTION")
    print("─" * 60)
    header = f"{'Metric':>10}  {'Real':>10}  {'1st-ord':>10}  {'2nd-ord':>10}"
    print(header)
    print("-" * len(header))
    for key in ["mean", "std", "min", "p25", "p50", "p75", "max"]:
        row = f"{key:>10}  "
        row += "  ".join(f"{stats[name][key]:>10.2f}" for name in ["real", "first", "second"])
        print(row)
    print()


def plot_length_distributions(data: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color, label_str in [("real",   REAL_COLOR,   "Real"),
                                    ("first",  FIRST_COLOR,  "1st-order syn"),
                                    ("second", SECOND_COLOR, "2nd-order syn")]:
        lengths = [len(s) for s in data[name]]
        ax.hist(lengths, bins=40, alpha=0.6, color=color, label=label_str, density=True)
    ax.set_xlabel("Session length (# actions)")
    ax.set_ylabel("Density")
    ax.set_title("Session Length Distributions")
    ax.legend()
    plt.tight_layout()
    plt.savefig("session_lengths.png", dpi=120)
    plt.show()
    print("  → Saved session_lengths.png\n")


# =============================================================================
# 5.  N-GRAM COMPARISON
# =============================================================================

def compute_ngrams(sessions: list, n: int) -> collections.Counter:
    counter = collections.Counter()
    for seq in sessions:
        for gram in zip(*[seq[i:] for i in range(n)]):
            counter[gram] += 1
    return counter


def ngram_top_overlap(real_counter, syn_counter, k: int = NGRAM_TOP_K) -> dict:
    top_real = set(dict(real_counter.most_common(k)).keys())
    top_syn  = set(dict(syn_counter.most_common(k)).keys())
    common   = top_real & top_syn
    return {
        "top_real_count": k,
        "top_syn_count":  k,
        "common":         len(common),
        "overlap_pct":    round(100 * len(common) / k, 1),
    }


def print_ngram_analysis(data: dict, index_to_action: dict):
    print("─" * 60)
    print("N-GRAM COMPARISON  (bigrams & trigrams)")
    print("─" * 60)
    for n, name in [(2, "Bigrams"), (3, "Trigrams")]:
        real_ng   = compute_ngrams(data["real"],   n)
        first_ng  = compute_ngrams(data["first"],  n)
        second_ng = compute_ngrams(data["second"], n)

        ol_first  = ngram_top_overlap(real_ng, first_ng,  NGRAM_TOP_K)
        ol_second = ngram_top_overlap(real_ng, second_ng, NGRAM_TOP_K)

        print(f"\n  {name} (top-{NGRAM_TOP_K} overlap with real):")
        print(f"    1st-order : {ol_first['common']}/{NGRAM_TOP_K}  "
              f"({ol_first['overlap_pct']}%)")
        print(f"    2nd-order : {ol_second['common']}/{NGRAM_TOP_K}  "
              f"({ol_second['overlap_pct']}%)")

        # Print top-5 real bigrams/trigrams
        print(f"\n  Top-5 real {name.lower()}:")
        for gram, cnt in real_ng.most_common(5):
            gram_str = " → ".join(label(a, index_to_action) for a in gram)
            in_first  = "✓" if gram in dict(first_ng.most_common(NGRAM_TOP_K))  else "✗"
            in_second = "✓" if gram in dict(second_ng.most_common(NGRAM_TOP_K)) else "✗"
            print(f"    [{gram_str}]  count={cnt}  1st={in_first}  2nd={in_second}")
    print()


# =============================================================================
# 6.  TRANSITION HOTSPOTS
# =============================================================================

def extract_top_transitions(sessions: list, k: int = TOP_K,
                             order: int = 1) -> list:
    """
    Return top-K bigram (order=1) or trigram (order=2) transitions as
    list of  ((from_tuple, to), count).
    """
    counter = collections.Counter()
    for seq in sessions:
        if order == 1:
            for a, b in zip(seq[:-1], seq[1:]):
                counter[(a,), b] += 1
        else:
            for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
                counter[(a, b), c] += 1
    return counter.most_common(k)


def print_transition_hotspots(data: dict, index_to_action: dict):
    print("─" * 60)
    print("TRANSITION HOTSPOTS")
    print("─" * 60)

    # ── First-order ──────────────────────────────────────────────────────────
    print(f"\n  TOP-{TOP_K} FIRST-ORDER TRANSITIONS  (a → b)\n")
    real_top1   = dict(extract_top_transitions(data["real"],   TOP_K, order=1))
    first_top1  = dict(extract_top_transitions(data["first"],  TOP_K, order=1))
    second_top1 = dict(extract_top_transitions(data["second"], TOP_K, order=1))

    header = f"  {'Transition':30}  {'Real':>6}  {'1st':>6}  {'2nd':>6}  Status"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_keys = sorted(set(real_top1) | set(first_top1) | set(second_top1),
                      key=lambda k: real_top1.get(k, 0), reverse=True)[:TOP_K]

    for key in all_keys:
        src, dst = key
        trans_str = " → ".join([label(s, index_to_action) for s in src] +
                                [label(dst, index_to_action)])
        r = real_top1.get(key, 0)
        f = first_top1.get(key, 0)
        s = second_top1.get(key, 0)
        status = ("MISSING_1ST " if f == 0 else "") + ("MISSING_2ND" if s == 0 else "")
        if not status:
            status = "OK"
        print(f"  {trans_str:30}  {r:>6}  {f:>6}  {s:>6}  {status}")

    # ── Second-order ─────────────────────────────────────────────────────────
    print(f"\n  TOP-{min(TOP_K, 15)} SECOND-ORDER TRANSITIONS  (a, b → c)\n")
    real_top2   = dict(extract_top_transitions(data["real"],   TOP_K, order=2))
    second_top2 = dict(extract_top_transitions(data["second"], TOP_K, order=2))

    keys2 = sorted(real_top2, key=real_top2.get, reverse=True)[:min(TOP_K, 15)]
    for key in keys2:
        src, dst = key
        trans_str = " → ".join([label(s, index_to_action) for s in src] +
                                [label(dst, index_to_action)])
        r = real_top2.get(key, 0)
        s = second_top2.get(key, 0)
        status = "MISSING_2ND" if s == 0 else "OK"
        print(f"  {trans_str:40}  Real={r:>5}  2nd={s:>5}  {status}")
    print()

    return {
        "real_top1":   {str(k): v for k, v in real_top1.items()},
        "first_top1":  {str(k): v for k, v in first_top1.items()},
        "second_top1": {str(k): v for k, v in second_top1.items()},
    }


# =============================================================================
# 7.  BEHAVIORAL INSIGHTS
# =============================================================================

def detect_loops(sessions: list, index_to_action: dict, top_n: int = 10) -> list:
    """Find A → B → A patterns (immediate return loops)."""
    loop_counter = collections.Counter()
    for seq in sessions:
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
            if a == c and a != b:
                loop_counter[(a, b)] += 1
    results = []
    for (a, b), cnt in loop_counter.most_common(top_n):
        results.append({
            "loop": f"{label(a, index_to_action)} → {label(b, index_to_action)} → {label(a, index_to_action)}",
            "count": cnt,
        })
    return results


def entry_exit_analysis(sessions: list, index_to_action: dict, top_n: int = 10) -> dict:
    """Which actions most often start or end a session?"""
    entry_counter = collections.Counter()
    exit_counter  = collections.Counter()
    for seq in sessions:
        if seq:
            entry_counter[seq[0]]  += 1
            exit_counter[seq[-1]]  += 1

    entries = [(label(a, index_to_action), cnt)
               for a, cnt in entry_counter.most_common(top_n)]
    exits   = [(label(a, index_to_action), cnt)
               for a, cnt in exit_counter.most_common(top_n)]
    return {"top_entries": entries, "top_exits": exits}


def dominant_paths(sessions: list, index_to_action: dict,
                   max_len: int = 4, top_n: int = 10) -> list:
    """Most frequent full sub-sequences up to max_len."""
    counter = collections.Counter()
    for seq in sessions:
        for length in range(2, min(max_len + 1, len(seq) + 1)):
            for start in range(len(seq) - length + 1):
                gram = tuple(seq[start:start + length])
                counter[gram] += 1
    results = []
    for gram, cnt in counter.most_common(top_n):
        path_str = " → ".join(label(a, index_to_action) for a in gram)
        results.append({"path": path_str, "count": cnt})
    return results


def generate_insights(data: dict, stats: dict, dists: dict,
                      jsd_results: dict, index_to_action: dict) -> list:
    """
    Compose a list of human-readable insight strings.
    """
    insights = []

    # ── Session length ────────────────────────────────────────────────────────
    real_mean  = stats["real"]["mean"]
    first_mean = stats["first"]["mean"]
    sec_mean   = stats["second"]["mean"]
    insights.append(
        f"Average session length – Real: {real_mean:.1f}, "
        f"1st-order: {first_mean:.1f}, 2nd-order: {sec_mean:.1f}."
    )
    if abs(sec_mean - real_mean) < abs(first_mean - real_mean):
        insights.append(
            "2nd-order sessions are closer to real session length than 1st-order sessions."
        )
    else:
        insights.append(
            "1st-order sessions are closer to real session length than 2nd-order sessions."
        )

    # ── Transition fidelity ──────────────────────────────────────────────────
    j1 = jsd_results["real_vs_first"]
    j2 = jsd_results["real_vs_second"]
    insights.append(
        f"Transition JSD – Real vs 1st-order: {j1:.4f}, Real vs 2nd-order: {j2:.4f}."
    )
    pct_improvement = 100 * (j1 - j2) / j1 if j1 > 0 else 0
    if j2 < j1:
        insights.append(
            f"2nd-order model reduces transition divergence by {pct_improvement:.1f}% "
            "vs 1st-order, indicating richer sequential structure."
        )
    else:
        insights.append(
            "1st-order model achieves lower transition divergence; "
            "2nd-order may be overfitting to training sequences."
        )

    # ── Action concentration ─────────────────────────────────────────────────
    real_probs = sorted(dists["real"].values(), reverse=True)
    top5_pct   = 100 * sum(real_probs[:5])
    insights.append(
        f"Top-5 actions account for {top5_pct:.1f}% of all real events "
        "(high action concentration)."
    )

    # ── Loops ────────────────────────────────────────────────────────────────
    real_loops = detect_loops(data["real"], index_to_action, top_n=3)
    if real_loops:
        loop_descs = ", ".join(f"\"{l['loop']}\" (×{l['count']})"
                               for l in real_loops[:3])
        insights.append(f"Most frequent navigation loops in real data: {loop_descs}.")

    # ── Entry / exit ─────────────────────────────────────────────────────────
    ee = entry_exit_analysis(data["real"], index_to_action, top_n=3)
    if ee["top_entries"]:
        top_entry = ee["top_entries"][0]
        insights.append(
            f"Most common session start: action '{top_entry[0]}' ({top_entry[1]} sessions)."
        )
    if ee["top_exits"]:
        top_exit = ee["top_exits"][0]
        insights.append(
            f"Most common session end: action '{top_exit[0]}' ({top_exit[1]} sessions)."
        )

    return insights


def print_behavioral_insights(data: dict, stats: dict, dists: dict,
                               jsd_results: dict, index_to_action: dict):
    print("─" * 60)
    print("BEHAVIORAL INSIGHTS")
    print("─" * 60)

    # Dominant paths
    print("\n  DOMINANT NAVIGATION PATHS (real sessions, sub-sequences up to 4):")
    for item in dominant_paths(data["real"], index_to_action, max_len=4, top_n=8):
        print(f"    [{item['count']:>5}×]  {item['path']}")

    # Loops
    print("\n  NAVIGATION LOOPS (A → B → A patterns, real sessions):")
    for item in detect_loops(data["real"], index_to_action, top_n=8):
        print(f"    [{item['count']:>5}×]  {item['loop']}")

    # Entry / exit
    print("\n  ENTRY / EXIT ACTIONS (real sessions):")
    ee = entry_exit_analysis(data["real"], index_to_action, top_n=5)
    print("    Entries:", ", ".join(f"{a}(×{c})" for a, c in ee["top_entries"]))
    print("    Exits  :", ", ".join(f"{a}(×{c})" for a, c in ee["top_exits"]))

    # High-level text insights
    insights = generate_insights(data, stats, dists, jsd_results, index_to_action)
    print("\n  SUMMARY INSIGHTS:")
    for i, txt in enumerate(insights, 1):
        print(f"    {i}. {txt}")
    print()

    return insights


# =============================================================================
# 8.  SAVE RESULTS
# =============================================================================

def save_results(action_dists, jsd_results, length_stats,
                 top_transitions, insights, path: Path = OUTPUT_JSON):
    def _make_serialisable(obj):
        """Recursively convert numpy types to native Python."""
        if isinstance(obj, dict):
            return {str(k): _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serialisable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    results = {
        "action_distribution": {
            "real":   {str(k): v for k, v in action_dists["real"].items()},
            "first":  {str(k): v for k, v in action_dists["first"].items()},
            "second": {str(k): v for k, v in action_dists["second"].items()},
        },
        "transition_divergence": jsd_results,
        "session_length_stats":  length_stats,
        "top_transitions":       top_transitions,
        "insights":              insights,
    }
    with open(path, "w") as f:
        json.dump(_make_serialisable(results), f, indent=2)
    print(f"  → Results saved to {path}\n")


# =============================================================================
# 9.  MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  PATTERN ANALYSIS: Real vs Synthetic Sessions")
    print("=" * 60 + "\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    data = load_all_data()
    all_actions    = data["all_actions"]
    index_to_action = data["index_to_action"]

    # ── 1. Action distributions ───────────────────────────────────────────────
    print("=" * 60)
    print("SECTION 1 — ACTION DISTRIBUTION")
    print("=" * 60)
    action_dists = {
        name: compute_action_distribution(data[name], all_actions)
        for name in ["real", "first", "second"]
    }
    print_action_distribution_summary(action_dists, top_n=15)
    if PLOT:
        plot_action_distributions(action_dists, all_actions, index_to_action)

    # ── 2. Transition matrices & JSD ─────────────────────────────────────────
    print("=" * 60)
    print("SECTION 2 — TRANSITION DISTRIBUTION")
    print("=" * 60)
    real_mat   = compute_transition_matrix(data["real"],   all_actions)
    first_mat  = compute_transition_matrix(data["first"],  all_actions)
    second_mat = compute_transition_matrix(data["second"], all_actions)
    jsd_results = print_transition_divergence(real_mat, first_mat, second_mat)
    if PLOT:
        plot_transition_heatmaps(real_mat, first_mat, second_mat,
                                 all_actions, index_to_action)

    # ── 3. Session length ─────────────────────────────────────────────────────
    print("=" * 60)
    print("SECTION 3 — SESSION LENGTH DISTRIBUTION")
    print("=" * 60)
    length_stats = {
        name: compute_length_stats(data[name])
        for name in ["real", "first", "second"]
    }
    print_length_stats(length_stats)
    if PLOT:
        plot_length_distributions(data)

    # ── 4. N-gram analysis ────────────────────────────────────────────────────
    print("=" * 60)
    print("SECTION 4 — N-GRAM COMPARISON")
    print("=" * 60)
    print_ngram_analysis(data, index_to_action)

    # ── 5. Transition hotspots ────────────────────────────────────────────────
    print("=" * 60)
    print("SECTION 5 — TRANSITION HOTSPOTS")
    print("=" * 60)
    top_transitions = print_transition_hotspots(data, index_to_action)

    # ── 6. Behavioral insights ────────────────────────────────────────────────
    print("=" * 60)
    print("SECTION 6 — BEHAVIORAL INSIGHTS")
    print("=" * 60)
    insights = print_behavioral_insights(
        data, length_stats, action_dists, jsd_results, index_to_action
    )

    # ── 7. Save ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    save_results(action_dists, jsd_results, length_stats,
                 top_transitions, insights)

    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()