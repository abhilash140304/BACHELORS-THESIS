"""
evaluate_models.py
------------------
Evaluates first-order and second-order Markov models on test sequences
using K-Fold Cross Validation.

Metrics:
  - Log-Likelihood
  - Perplexity
  - Unseen Transitions
  - KL Divergence (empirical vs model)

Author: Bachelor's Thesis — Probabilistic Modeling of User Interaction Sequences
"""

import json
import os
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

EPSILON = 1e-10          # Replaces zero probabilities in log / KL computation

BASE_TRAINED_DIR  = r"C:\Users\shaik\Thesis Batchlor's\outputs_of_trained_data"
BASE_SPLIT_DIR    = r"C:\Users\shaik\Thesis Batchlor's\split_data"
OUTPUT_BASE_DIR   = "outputs_after_model_evaluation"


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(fold_id: int) -> dict:
    """
    Load all model artefacts and test data for a specific fold.

    Args:
        fold_id: Integer fold identifier (e.g. 1, 2, … K)

    Returns a dict with:
      - action_to_index    : {str -> int}
      - index_to_action    : {str -> str}
      - first_order_matrix  : np.ndarray (N, N)
      - second_order_tensor : np.ndarray (N, N, N)
      - first_order_counts  : np.ndarray (N, N)
      - second_order_counts : np.ndarray (N, N, N)
      - test_sequences      : list[list[str]]
    """
    trained_dir = os.path.join(BASE_TRAINED_DIR, f"output_fold_{fold_id}")
    split_dir   = os.path.join(BASE_SPLIT_DIR,   f"fold_{fold_id}")

    print(f"\n{'=' * 60}")
    print(f"  Loading data for fold {fold_id} …")
    print(f"{'=' * 60}")

    # ── mappings ──────────────────────────────────────────────────
    with open(os.path.join(trained_dir, "action_to_index.json"), "r") as f:
        action_to_index: dict[str, int] = json.load(f)

    with open(os.path.join(trained_dir, "index_to_action.json"), "r") as f:
        index_to_action: dict[str, str] = json.load(f)

    n_actions = len(action_to_index)
    print(f"  Vocabulary size           : {n_actions} actions")

    # ── transition matrices ───────────────────────────────────────
    first_order_matrix  = np.load(os.path.join(trained_dir, "first_order_transition_matrix.npy"))
    second_order_tensor = np.load(os.path.join(trained_dir, "second_order_transition_tensor.npy"))
    first_order_counts  = np.load(os.path.join(trained_dir, "first_order_counts.npy"))
    second_order_counts = np.load(os.path.join(trained_dir, "second_order_counts.npy"))

    # ── shape validation ──────────────────────────────────────────
    expected_fo = (n_actions, n_actions)
    expected_so = (n_actions, n_actions, n_actions)

    if first_order_matrix.shape != expected_fo:
        raise ValueError(
            f"first_order_matrix shape mismatch: "
            f"expected {expected_fo}, got {first_order_matrix.shape}"
        )
    if second_order_tensor.shape != expected_so:
        raise ValueError(
            f"second_order_tensor shape mismatch: "
            f"expected {expected_so}, got {second_order_tensor.shape}"
        )

    print(f"  First-order matrix shape  : {first_order_matrix.shape}")
    print(f"  Second-order tensor shape : {second_order_tensor.shape}")

    # ── test sequences ────────────────────────────────────────────
    with open(os.path.join(split_dir, "test_sequences.json"), "r") as f:
        test_sequences: list[list[str]] = json.load(f)

    print(f"  Test sequences loaded     : {len(test_sequences)}")

    return {
        "action_to_index":    action_to_index,
        "index_to_action":    index_to_action,
        "first_order_matrix":  first_order_matrix,
        "second_order_tensor": second_order_tensor,
        "first_order_counts":  first_order_counts,
        "second_order_counts": second_order_counts,
        "test_sequences":      test_sequences,
    }


# ─────────────────────────────────────────────
# 2. EMPIRICAL DISTRIBUTION BUILDERS
# ─────────────────────────────────────────────

def build_empirical_first_order(
    test_sequences: list[list[str]],
    action_to_index: dict[str, int],
    n_actions: int,
) -> np.ndarray:
    """
    Build an empirical first-order transition matrix from test sequences.

    Each row i is normalised to sum to 1 (rows with no observations remain zero).

    Returns:
        emp_matrix : np.ndarray of shape (N, N)
    """
    counts = np.zeros((n_actions, n_actions), dtype=np.float64)

    for seq in test_sequences:
        if len(seq) < 2:
            continue
        for t in range(len(seq) - 1):
            a, b = seq[t], seq[t + 1]
            if a in action_to_index and b in action_to_index:
                counts[action_to_index[a], action_to_index[b]] += 1.0

    # Row-normalise
    row_sums = counts.sum(axis=1, keepdims=True)
    emp_matrix = np.where(row_sums > 0, counts / row_sums, 0.0)
    return emp_matrix


def build_empirical_second_order(
    test_sequences: list[list[str]],
    action_to_index: dict[str, int],
    n_actions: int,
) -> np.ndarray:
    """
    Build an empirical second-order transition tensor from test sequences.

    Each slice [h, i, :] is normalised to sum to 1.

    Returns:
        emp_tensor : np.ndarray of shape (N, N, N)
    """
    counts = np.zeros((n_actions, n_actions, n_actions), dtype=np.float64)

    for seq in test_sequences:
        if len(seq) < 3:
            continue
        for t in range(1, len(seq) - 1):
            a, b, c = seq[t - 1], seq[t], seq[t + 1]
            if a in action_to_index and b in action_to_index and c in action_to_index:
                counts[action_to_index[a], action_to_index[b], action_to_index[c]] += 1.0

    # Slice-normalise over last axis
    slice_sums = counts.sum(axis=2, keepdims=True)
    emp_tensor = np.where(slice_sums > 0, counts / slice_sums, 0.0)
    return emp_tensor


# ─────────────────────────────────────────────
# 3. KL DIVERGENCE
# ─────────────────────────────────────────────

def compute_kl_divergence_first_order(
    empirical_matrix: np.ndarray,
    model_matrix: np.ndarray,
) -> float:
    """
    Compute KL Divergence between empirical and model first-order distributions.

    KL(P_emp || P_model) = Σ_{i,j} P_emp(i,j) * log[ P_emp(i,j) / P_model(i,j) ]

    Only rows where the empirical distribution is non-zero are included
    (i.e. rows observed in the test set).  Epsilon smoothing prevents log(0).

    Args:
        empirical_matrix : shape (N, N) — row-normalised empirical counts
        model_matrix     : shape (N, N) — model transition probabilities

    Returns:
        Scalar KL divergence (float)
    """
    kl_total = 0.0

    n = empirical_matrix.shape[0]
    for i in range(n):
        # Only consider rows that appear in the empirical distribution
        if empirical_matrix[i].sum() == 0:
            continue

        p = empirical_matrix[i]      # empirical distribution over next actions
        q = model_matrix[i]          # model distribution over next actions

        # KL contribution for this conditioning context
        # Only sum over j where p_j > 0 to avoid 0 * log(0) = NaN
        mask = p > 0
        kl_total += np.sum(
            p[mask] * np.log((p[mask] + EPSILON) / (q[mask] + EPSILON))
        )

    return float(kl_total)


def compute_kl_divergence_second_order(
    empirical_tensor: np.ndarray,
    model_tensor: np.ndarray,
) -> float:
    """
    Compute KL Divergence between empirical and model second-order distributions.

    KL(P_emp || P_model) = Σ_{h,i,j} P_emp(h,i,j) * log[ P_emp(h,i,j) / P_model(h,i,j) ]

    Only slices (h, i) where the empirical distribution is non-zero are included.

    Args:
        empirical_tensor : shape (N, N, N) — slice-normalised empirical counts
        model_tensor     : shape (N, N, N) — model transition probabilities

    Returns:
        Scalar KL divergence (float)
    """
    kl_total = 0.0

    n = empirical_tensor.shape[0]
    for h in range(n):
        for i in range(n):
            if empirical_tensor[h, i].sum() == 0:
                continue

            p = empirical_tensor[h, i]
            q = model_tensor[h, i]

            mask = p > 0
            kl_total += np.sum(
                p[mask] * np.log((p[mask] + EPSILON) / (q[mask] + EPSILON))
            )

    return float(kl_total)


# ─────────────────────────────────────────────
# 4. FIRST-ORDER EVALUATION
# ─────────────────────────────────────────────

def evaluate_first_order(
    test_sequences: list[list[str]],
    transition_matrix: np.ndarray,
    action_to_index: dict[str, int],
) -> dict:
    """
    Evaluate the first-order Markov model.

    For each consecutive pair (a_t, a_{t+1}) in every sequence:
      - Look up P(a_{t+1} | a_t) from transition_matrix[i, j]
      - Accumulate log-probability (using epsilon for zeros)
      - Count unseen transitions (raw probability == 0)

    Args:
        test_sequences    : list of action-label sequences
        transition_matrix : shape (N, N), rows are conditioning actions
        action_to_index   : maps action string to integer index

    Returns:
        dict with all evaluation metrics including KL divergence
    """
    total_log_likelihood = 0.0
    total_transitions    = 0
    unseen_transitions   = 0
    n_actions            = len(action_to_index)

    for seq in test_sequences:
        if len(seq) < 2:
            continue

        for t in range(len(seq) - 1):
            current_action = seq[t]
            next_action    = seq[t + 1]

            if current_action not in action_to_index or next_action not in action_to_index:
                continue

            i = action_to_index[current_action]
            j = action_to_index[next_action]

            prob = transition_matrix[i, j]

            if prob == 0.0:
                unseen_transitions += 1

            total_log_likelihood += np.log(prob + EPSILON)
            total_transitions    += 1

    perplexity = (
        float("inf")
        if total_transitions == 0
        else float(np.exp(-total_log_likelihood / total_transitions))
    )

    unseen_ratio = (
        unseen_transitions / total_transitions if total_transitions > 0 else 0.0
    )

    # ── KL Divergence ──────────────────────────────────────────────
    empirical_matrix = build_empirical_first_order(test_sequences, action_to_index, n_actions)
    kl_divergence    = compute_kl_divergence_first_order(empirical_matrix, transition_matrix)

    return {
        "log_likelihood":     float(total_log_likelihood),
        "total_transitions":  int(total_transitions),
        "perplexity":         perplexity,
        "unseen_transitions": int(unseen_transitions),
        "unseen_ratio":       float(unseen_ratio),
        "kl_divergence":      kl_divergence,
    }


# ─────────────────────────────────────────────
# 5. SECOND-ORDER EVALUATION
# ─────────────────────────────────────────────

def evaluate_second_order(
    test_sequences: list[list[str]],
    transition_tensor: np.ndarray,
    action_to_index: dict[str, int],
) -> dict:
    """
    Evaluate the second-order Markov model.

    For each triple (a_{t-1}, a_t, a_{t+1}) in every sequence:
      - Look up P(a_{t+1} | a_{t-1}, a_t) from transition_tensor[h, i, j]
      - Accumulate log-probability (using epsilon for zeros)
      - Count unseen transitions (raw probability == 0)

    Args:
        test_sequences    : list of action-label sequences
        transition_tensor : shape (N, N, N), [prev, current, next]
        action_to_index   : maps action string to integer index

    Returns:
        dict with all evaluation metrics including KL divergence
    """
    total_log_likelihood = 0.0
    total_transitions    = 0
    unseen_transitions   = 0
    n_actions            = len(action_to_index)

    for seq in test_sequences:
        if len(seq) < 3:
            continue

        for t in range(1, len(seq) - 1):
            prev_action    = seq[t - 1]
            current_action = seq[t]
            next_action    = seq[t + 1]

            if (
                prev_action    not in action_to_index
                or current_action not in action_to_index
                or next_action    not in action_to_index
            ):
                continue

            h = action_to_index[prev_action]
            i = action_to_index[current_action]
            j = action_to_index[next_action]

            prob = transition_tensor[h, i, j]

            if prob == 0.0:
                unseen_transitions += 1

            total_log_likelihood += np.log(prob + EPSILON)
            total_transitions    += 1

    perplexity = (
        float("inf")
        if total_transitions == 0
        else float(np.exp(-total_log_likelihood / total_transitions))
    )

    unseen_ratio = (
        unseen_transitions / total_transitions if total_transitions > 0 else 0.0
    )

    # ── KL Divergence ──────────────────────────────────────────────
    empirical_tensor = build_empirical_second_order(test_sequences, action_to_index, n_actions)
    kl_divergence    = compute_kl_divergence_second_order(empirical_tensor, transition_tensor)

    return {
        "log_likelihood":     float(total_log_likelihood),
        "total_transitions":  int(total_transitions),
        "perplexity":         perplexity,
        "unseen_transitions": int(unseen_transitions),
        "unseen_ratio":       float(unseen_ratio),
        "kl_divergence":      kl_divergence,
    }


# ─────────────────────────────────────────────
# 6. SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(results: dict, filepath: str) -> None:
    """Serialise a results dict to a JSON file, converting numpy types."""
    serialisable = {}
    for k, v in results.items():
        if isinstance(v, (np.floating, np.integer)):
            serialisable[k] = float(v)
        elif isinstance(v, dict):
            # Recursively handle nested dicts (used in summary)
            serialisable[k] = {
                kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                for kk, vv in v.items()
            }
        else:
            serialisable[k] = v

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(serialisable, f, indent=4)
    print(f"  Saved → {filepath}")


# ─────────────────────────────────────────────
# 7. CROSS-VALIDATION AGGREGATION
# ─────────────────────────────────────────────

AGGREGATED_METRICS = ["log_likelihood", "perplexity", "unseen_ratio", "kl_divergence"]


def aggregate_cv_results(fold_results: list[dict]) -> dict:
    """
    Compute mean and standard deviation for each metric across folds.

    Args:
        fold_results : list of per-fold result dicts (one per fold)

    Returns:
        dict mapping metric_name -> {"mean": float, "std": float}
    """
    aggregated = {}
    for metric in AGGREGATED_METRICS:
        values = [r[metric] for r in fold_results if metric in r]
        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
        }
    return aggregated


def build_summary(
    fo_fold_results: list[dict],
    so_fold_results: list[dict],
    n_folds: int,
) -> dict:
    """
    Build the summary dict that is saved to summary.json.

    Includes per-metric mean/std for each model plus a model comparison.
    """
    fo_agg = aggregate_cv_results(fo_fold_results)
    so_agg = aggregate_cv_results(so_fold_results)

    # ── model comparison (mean values) ────────────────────────────
    comparison = {
        "higher_log_likelihood": (
            "first_order"
            if fo_agg["log_likelihood"]["mean"] >= so_agg["log_likelihood"]["mean"]
            else "second_order"
        ),
        "lower_perplexity": (
            "first_order"
            if fo_agg["perplexity"]["mean"] <= so_agg["perplexity"]["mean"]
            else "second_order"
        ),
        "lower_unseen_ratio": (
            "first_order"
            if fo_agg["unseen_ratio"]["mean"] <= so_agg["unseen_ratio"]["mean"]
            else "second_order"
        ),
        "lower_kl_divergence": (
            "first_order"
            if fo_agg["kl_divergence"]["mean"] <= so_agg["kl_divergence"]["mean"]
            else "second_order"
        ),
    }

    return {
        "n_folds":       n_folds,
        "first_order":   fo_agg,
        "second_order":  so_agg,
        "model_comparison": comparison,
    }


# ─────────────────────────────────────────────
# 8. PRINT SUMMARY
# ─────────────────────────────────────────────

def print_fold_summary(fold_id: int, fo_results: dict, so_results: dict) -> None:
    """Print a formatted per-fold comparison of both model evaluations."""
    sep = "─" * 60

    print(f"\n{'=' * 60}")
    print(f"  FOLD {fold_id} — EVALUATION RESULTS")
    print(f"{'=' * 60}")

    for label, res in [
        ("First-Order Markov",  fo_results),
        ("Second-Order Markov", so_results),
    ]:
        print(f"\n  {label}")
        print(f"  {sep}")
        print(f"    Log-Likelihood      : {res['log_likelihood']:.4f}")
        print(f"    Total Transitions   : {res['total_transitions']}")
        print(f"    Perplexity          : {res['perplexity']:.4f}")
        print(f"    Unseen Transitions  : {res['unseen_transitions']}")
        print(f"    Unseen Ratio        : {res['unseen_ratio']:.4%}")
        print(f"    KL Divergence       : {res['kl_divergence']:.6f}")


def print_cv_summary(summary: dict) -> None:
    """Print the aggregated cross-validation summary across all folds."""
    sep = "─" * 60
    n   = summary["n_folds"]

    print(f"\n{'=' * 60}")
    print(f"  CROSS-VALIDATION SUMMARY  ({n} folds)")
    print(f"{'=' * 60}")

    for label, key in [
        ("First-Order Markov",  "first_order"),
        ("Second-Order Markov", "second_order"),
    ]:
        agg = summary[key]
        print(f"\n  {label}")
        print(f"  {sep}")
        for metric in AGGREGATED_METRICS:
            m = agg[metric]["mean"]
            s = agg[metric]["std"]
            if metric in ("unseen_ratio",):
                print(f"    {metric:<22}: {m:.4%}  ± {s:.4%}")
            elif metric in ("kl_divergence",):
                print(f"    {metric:<22}: {m:.6f}  ± {s:.6f}")
            elif metric in ("log_likelihood",):
                print(f"    {metric:<22}: {m:.4f}  ± {s:.4f}")
            else:
                print(f"    {metric:<22}: {m:.4f}  ± {s:.4f}")

    # ── model comparison ──────────────────────────────────────────
    cmp = summary["model_comparison"]
    print(f"\n{'=' * 60}")
    print("  MODEL COMPARISON  (mean across folds)")
    print(f"{'=' * 60}")
    print(f"\n  Higher Log-Likelihood  → {cmp['higher_log_likelihood']}")
    print(f"  Lower Perplexity       → {cmp['lower_perplexity']}")
    print(f"  Lower Unseen Ratio     → {cmp['lower_unseen_ratio']}")
    print(f"  Lower KL Divergence    → {cmp['lower_kl_divergence']}")
    print(f"\n{'=' * 60}")


# ─────────────────────────────────────────────
# 9. K-FOLD CROSS VALIDATION RUNNER
# ─────────────────────────────────────────────

def detect_n_folds() -> int:
    """
    Auto-detect the number of folds by scanning split_data/ for fold_X directories.

    Returns:
        Number of folds found (raises RuntimeError if none found)
    """
    split_path = Path(BASE_SPLIT_DIR)
    fold_dirs  = sorted(split_path.glob("fold_*"))
    valid_folds = [
        d for d in fold_dirs
        if d.is_dir() and d.name.replace("fold_", "").isdigit()
    ]
    if not valid_folds:
        raise RuntimeError(
            f"No fold directories found under '{BASE_SPLIT_DIR}'. "
            "Expected structure: fold_1/, fold_2/, …"
        )
    n = len(valid_folds)
    print(f"\nDetected {n} fold(s) in '{BASE_SPLIT_DIR}'")
    return n


def run_cross_validation(n_folds: int | None = None) -> dict:
    """
    Run K-Fold cross validation over all available folds.

    For each fold:
      1. Load fold-specific model artefacts and test sequences
      2. Evaluate first-order model  → metrics + KL divergence
      3. Evaluate second-order model → metrics + KL divergence
      4. Save per-fold JSONs to outputs_after_model_evaluation/fold_X/
    Then:
      5. Aggregate results (mean ± std) across folds
      6. Save summary.json
      7. Print CV summary

    Args:
        n_folds : number of folds to iterate; auto-detected if None

    Returns:
        summary dict (same structure as saved summary.json)
    """
    if n_folds is None:
        n_folds = detect_n_folds()

    fo_all_results: list[dict] = []
    so_all_results: list[dict] = []

    for fold_id in range(1, n_folds + 1):
        print(f"\n{'#' * 60}")
        print(f"  FOLD {fold_id} / {n_folds}")
        print(f"{'#' * 60}")

        # ── load ──────────────────────────────────────────────────
        data = load_data(fold_id)

        action_to_index   = data["action_to_index"]
        test_sequences    = data["test_sequences"]
        first_order_mat   = data["first_order_matrix"]
        second_order_tens = data["second_order_tensor"]

        # ── evaluate ──────────────────────────────────────────────
        print(f"\n  Evaluating First-Order  (fold {fold_id}) …")
        fo_results = evaluate_first_order(test_sequences, first_order_mat,   action_to_index)

        print(f"  Evaluating Second-Order (fold {fold_id}) …")
        so_results = evaluate_second_order(test_sequences, second_order_tens, action_to_index)

        fo_all_results.append(fo_results)
        so_all_results.append(so_results)

        # ── save per-fold results ──────────────────────────────────
        fold_out_dir = os.path.join(OUTPUT_BASE_DIR, f"fold_{fold_id}")
        print(f"\n  Saving fold {fold_id} results …")
        save_results(fo_results, os.path.join(fold_out_dir, "first_order.json"))
        save_results(so_results, os.path.join(fold_out_dir, "second_order.json"))

        # ── per-fold console summary ───────────────────────────────
        print_fold_summary(fold_id, fo_results, so_results)

    # ── aggregate and save summary ─────────────────────────────────
    summary = build_summary(fo_all_results, so_all_results, n_folds)
    summary_path = os.path.join(OUTPUT_BASE_DIR, "summary.json")
    print(f"\n{'=' * 60}")
    print("  Saving cross-validation summary …")
    save_results(summary, summary_path)

    # ── print final CV summary ─────────────────────────────────────
    print_cv_summary(summary)

    return summary


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────

def main() -> None:
    """
    Entry point.

    Auto-detects the number of folds and runs full K-Fold cross validation
    for both first-order and second-order Markov models.
    """
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    run_cross_validation()


if __name__ == "__main__":
    main()