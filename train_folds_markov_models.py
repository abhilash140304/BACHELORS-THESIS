"""
Markov Models for User Session Sequences  —  K-Fold Edition
=============================================================
Trains first-order and second-order Markov models for each cross-validation
fold found inside `split_data/`.

Expected input layout
---------------------
split_data/
    fold_1/
        train_sequences.json
        test_sequences.json
    fold_2/  ...
    fold_N/  ...

Output layout
-------------
outputs_of_trained_data/
    output_fold_1/
        first_order_counts.npy
        first_order_transition_matrix.npy
        first_order_transition_matrix.csv
        second_order_counts.npy
        second_order_transition_tensor.npy
        action_to_index.json
        index_to_action.json
        model_info.json
    output_fold_2/  ...
    output_fold_N/  ...
    cross_fold_summary.json          ← aggregated stats across all folds
"""

import json
import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. FOLD DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_folds(split_data_dir: str) -> list[tuple[str, str, str]]:
    """
    Scan `split_data_dir` for sub-directories named fold_* and return one
    entry per fold with the fold name and paths to its two JSON files.

    Args:
        split_data_dir: Root directory that contains fold_* sub-folders.

    Returns:
        List of (fold_name, train_path, test_path) tuples, sorted by fold name.

    Raises:
        FileNotFoundError: If no valid fold directories are found.
    """
    entries = []
    for name in sorted(os.listdir(split_data_dir)):
        fold_dir = os.path.join(split_data_dir, name)
        if not (os.path.isdir(fold_dir) and name.startswith("fold_")):
            continue
        train_path = os.path.join(fold_dir, "train_sequences.json")
        test_path  = os.path.join(fold_dir, "test_sequences.json")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            print(f"  [WARN] '{name}' is missing train/test JSON — skipped.")
            continue
        entries.append((name, train_path, test_path))

    if not entries:
        raise FileNotFoundError(
            f"No valid fold_* directories found inside '{split_data_dir}'."
        )
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str, fold_name: str) -> tuple[list, list]:
    """
    Load train and test sequences from JSON files for a single fold.

    Args:
        train_path: Path to train_sequences.json.
        test_path:  Path to test_sequences.json.
        fold_name:  Human-readable label used in console output.

    Returns:
        (train_sequences, test_sequences) — each a list of action lists.
    """
    with open(train_path, "r") as f:
        train_sequences = json.load(f)
    with open(test_path, "r") as f:
        test_sequences = json.load(f)

    train_lengths     = [len(s) for s in train_sequences]
    total_transitions = sum(max(len(s) - 1, 0) for s in train_sequences)

    print(f"\n  Train sequences      : {len(train_sequences):,}")
    print(f"  Test  sequences      : {len(test_sequences):,}")
    print(f"  Avg train seq length : {np.mean(train_lengths):.2f}")
    print(f"  Min / Max length     : {min(train_lengths)} / {max(train_lengths)}")
    print(f"  Total transitions    : {total_transitions:,}")

    return train_sequences, test_sequences


# ─────────────────────────────────────────────────────────────────────────────
# 3. ACTION MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def build_action_mapping(sequences: list) -> tuple[dict, dict]:
    """
    Extract unique actions from sequences and create bidirectional index maps.

    Args:
        sequences: List of action sequences (training data for one fold).

    Returns:
        (action_to_index, index_to_action) dictionaries.
    """
    unique_actions  = sorted({action for seq in sequences for action in seq})
    action_to_index = {action: idx for idx, action in enumerate(unique_actions)}
    index_to_action = {idx: action for action, idx in action_to_index.items()}

    preview = unique_actions[:10]
    suffix  = "..." if len(unique_actions) > 10 else ""
    print(f"\n  Unique actions found : {len(unique_actions)}")
    print(f"  Actions              : {preview}{suffix}")

    return action_to_index, index_to_action


# ─────────────────────────────────────────────────────────────────────────────
# 4. FIRST-ORDER MARKOV MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_first_order_model(
    sequences: list,
    action_to_index: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a first-order Markov model: P(a_t+1 | a_t).

    Steps:
      1. Count transitions a_t → a_t+1.
      2. Apply Laplace (add-1) smoothing.
      3. Row-normalise to obtain probabilities.

    Args:
        sequences:       Training sequences for one fold.
        action_to_index: Action → integer index mapping.

    Returns:
        (counts, transition_matrix) — shapes (k, k) each.
        `counts` holds raw transition counts; `transition_matrix` holds
        smoothed, row-normalised probabilities.
    """
    k      = len(action_to_index)
    counts = np.zeros((k, k), dtype=np.float64)

    skipped = 0
    for seq in sequences:
        if len(seq) < 2:          # need at least one transition
            skipped += 1
            continue
        for t in range(len(seq) - 1):
            i = action_to_index[seq[t]]
            j = action_to_index[seq[t + 1]]
            counts[i, j] += 1

    smoothed          = counts + 1.0                          # Laplace smoothing
    row_sums          = smoothed.sum(axis=1, keepdims=True)
    transition_matrix = smoothed / row_sums                   # row-normalise → P

    print(f"\n  [1st-order] Transitions counted : {int(counts.sum()):,}")
    print(f"  [1st-order] Sequences skipped   : {skipped}")

    return counts, transition_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 5. SECOND-ORDER MARKOV MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_second_order_model(
    sequences: list,
    action_to_index: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a second-order Markov model: P(a_t+1 | a_t-1, a_t).

    Steps:
      1. Count transitions (a_t-1, a_t) → a_t+1.
      2. Apply Laplace (add-1) smoothing.
      3. Normalise over the last axis to obtain probabilities.

    Args:
        sequences:       Training sequences for one fold.
        action_to_index: Action → integer index mapping.

    Returns:
        (counts, transition_tensor) — shapes (k, k, k) each.
        `counts` holds raw counts; `transition_tensor` holds smoothed,
        normalised probabilities.
    """
    k      = len(action_to_index)
    counts = np.zeros((k, k, k), dtype=np.float64)

    skipped = 0
    for seq in sequences:
        if len(seq) < 3:          # need at least two context steps
            skipped += 1
            continue
        for t in range(len(seq) - 2):
            i = action_to_index[seq[t]]
            j = action_to_index[seq[t + 1]]
            m = action_to_index[seq[t + 2]]
            counts[i, j, m] += 1

    smoothed          = counts + 1.0                          # Laplace smoothing
    sums              = smoothed.sum(axis=2, keepdims=True)
    transition_tensor = smoothed / sums                       # normalise over next-action axis

    print(f"\n  [2nd-order] Transitions counted : {int(counts.sum()):,}")
    print(f"  [2nd-order] Sequences skipped   : {skipped}")

    return counts, transition_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE OUTPUTS  (per fold)
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    output_dir: str,
    fold_name: str,
    action_to_index: dict,
    index_to_action: dict,
    fo_counts: np.ndarray,
    fo_matrix: np.ndarray,
    so_counts: np.ndarray,
    so_tensor: np.ndarray,
    train_sequences: list,
) -> dict:
    """
    Persist all model artefacts for one fold to `output_dir`.

    Files written
    -------------
    first_order_counts.npy
    first_order_transition_matrix.npy
    first_order_transition_matrix.csv
    second_order_counts.npy
    second_order_transition_tensor.npy
    action_to_index.json
    index_to_action.json
    model_info.json

    Args:
        output_dir:       Fold-specific output directory (created if absent).
        fold_name:        Label for this fold (e.g. "fold_1").
        action_to_index:  Action → index mapping.
        index_to_action:  Index → action mapping.
        fo_counts:        First-order raw counts  (k, k).
        fo_matrix:        First-order probabilities (k, k).
        so_counts:        Second-order raw counts  (k, k, k).
        so_tensor:        Second-order probabilities (k, k, k).
        train_sequences:  Training sequences (used for metadata).

    Returns:
        model_info dict (also written to model_info.json).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── numpy arrays ──────────────────────────────────────────────────────────
    np.save(os.path.join(output_dir, "first_order_counts.npy"),            fo_counts)
    np.save(os.path.join(output_dir, "first_order_transition_matrix.npy"), fo_matrix)
    np.save(os.path.join(output_dir, "second_order_counts.npy"),           so_counts)
    np.save(os.path.join(output_dir, "second_order_transition_tensor.npy"), so_tensor)

    # ── CSV of first-order transition matrix ──────────────────────────────────
    actions = list(action_to_index.keys())
    df = pd.DataFrame(fo_matrix, index=actions, columns=actions)
    df.index.name = "from \\ to"
    df.to_csv(
        os.path.join(output_dir, "first_order_transition_matrix.csv"),
        float_format="%.6f",
    )

    # ── action ↔ index JSON mappings ─────────────────────────────────────────
    with open(os.path.join(output_dir, "action_to_index.json"), "w") as f:
        json.dump(action_to_index, f, indent=2)

    index_to_action_str = {str(k): v for k, v in index_to_action.items()}
    with open(os.path.join(output_dir, "index_to_action.json"), "w") as f:
        json.dump(index_to_action_str, f, indent=2)

    # ── model_info.json ───────────────────────────────────────────────────────
    model_info = {
        "fold": fold_name,
        "num_actions": len(action_to_index),
        "smoothing": "Laplace (add-1)",
        "first_order": {
            "total_transitions": int(fo_counts.sum()),
            "matrix_shape": list(fo_matrix.shape),
        },
        "second_order": {
            "total_transitions": int(so_counts.sum()),
            "tensor_shape": list(so_tensor.shape),
        },
        "training_sequences": len(train_sequences),
        "avg_sequence_length": round(
            float(np.mean([len(s) for s in train_sequences])), 4
        ),
    }
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n  Outputs saved → '{output_dir}/'")
    return model_info


# ─────────────────────────────────────────────────────────────────────────────
# 7. DIAGNOSTICS  (per fold)
# ─────────────────────────────────────────────────────────────────────────────

def print_diagnostics(
    fo_matrix: np.ndarray,
    action_to_index: dict,
    index_to_action: dict,
    preview_size: int = 5,
) -> None:
    """
    Print a corner preview of the transition matrix and the top-5 transitions.

    Args:
        fo_matrix:        First-order transition probability matrix (k, k).
        action_to_index:  Action → index mapping.
        index_to_action:  Index → action mapping.
        preview_size:     Corner size for matrix preview.
    """
    actions = list(action_to_index.keys())
    n = min(preview_size, len(actions))

    print("\n  ── Transition Matrix Preview  (first {}×{} corner) ──".format(n, n))
    df_preview = pd.DataFrame(
        fo_matrix[:n, :n],
        index=actions[:n],
        columns=actions[:n],
    )
    print(df_preview.to_string(float_format=lambda x: f"{x:.4f}"))

    # Top-5 non-self-loop transitions
    print("\n  ── Top 5 Most Probable Transitions ──")
    k = fo_matrix.shape[0]
    flat_order  = np.argsort(fo_matrix, axis=None)[::-1]
    rows, cols  = np.unravel_index(flat_order, (k, k))
    shown = 0
    for r, c in zip(rows, cols):
        if r == c:
            continue
        print(
            f"    {index_to_action[r]!r:>15} → {index_to_action[c]!r:<15}"
            f"  P = {fo_matrix[r, c]:.4f}"
        )
        shown += 1
        if shown >= 5:
            break


# ─────────────────────────────────────────────────────────────────────────────
# 8. CROSS-FOLD SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def save_cross_fold_summary(all_info: list[dict], base_output_dir: str) -> None:
    """
    Aggregate model_info dicts from every fold into a single summary JSON.

    Writes  outputs_of_trained_data/cross_fold_summary.json

    Args:
        all_info:        List of model_info dicts, one per fold.
        base_output_dir: Root output directory.
    """
    summary = {
        "total_folds": len(all_info),
        "folds": all_info,
        "aggregate": {
            "avg_num_actions": round(
                float(np.mean([d["num_actions"] for d in all_info])), 2
            ),
            "avg_training_sequences": round(
                float(np.mean([d["training_sequences"] for d in all_info])), 2
            ),
            "avg_sequence_length": round(
                float(np.mean([d["avg_sequence_length"] for d in all_info])), 4
            ),
            "avg_first_order_transitions": round(
                float(np.mean([d["first_order"]["total_transitions"] for d in all_info])), 2
            ),
            "avg_second_order_transitions": round(
                float(np.mean([d["second_order"]["total_transitions"] for d in all_info])), 2
            ),
        },
    }
    os.makedirs(base_output_dir, exist_ok=True)
    path = os.path.join(base_output_dir, "cross_fold_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Cross-fold summary saved → '{path}'")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SINGLE-FOLD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_fold(
    fold_name: str,
    train_path: str,
    test_path: str,
    base_output_dir: str,
) -> dict:
    """
    Run the full train → save → diagnose pipeline for one fold.

    Args:
        fold_name:       e.g. "fold_1"
        train_path:      Path to train_sequences.json for this fold.
        test_path:       Path to test_sequences.json for this fold.
        base_output_dir: Root output directory (e.g. "outputs_of_trained_data").

    Returns:
        model_info dict for this fold.
    """
    # e.g.  outputs_of_trained_data/output_fold_1/
    fold_output_dir = os.path.join(base_output_dir, f"output_{fold_name}")

    train_sequences, test_sequences = load_data(train_path, test_path, fold_name)
    action_to_index, index_to_action = build_action_mapping(train_sequences)

    fo_counts, fo_matrix = build_first_order_model(train_sequences, action_to_index)
    so_counts, so_tensor  = build_second_order_model(train_sequences, action_to_index)

    model_info = save_outputs(
        fold_output_dir,
        fold_name,
        action_to_index,
        index_to_action,
        fo_counts, fo_matrix,
        so_counts, so_tensor,
        train_sequences,
    )

    print_diagnostics(fo_matrix, action_to_index, index_to_action)

    return model_info


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SPLIT_DATA_DIR  = "split_data"             # contains fold_1/ … fold_N/
    BASE_OUTPUT_DIR = "outputs_of_trained_data" # output root

    # ── discover folds ────────────────────────────────────────────────────────
    folds = discover_folds(SPLIT_DATA_DIR)

    print("=" * 60)
    print(f"  Found {len(folds)} fold(s): {[f[0] for f in folds]}")
    print("=" * 60)

    # ── process every fold ────────────────────────────────────────────────────
    all_model_info = []
    for fold_name, train_path, test_path in folds:
        print("\n" + "=" * 60)
        print(f"  PROCESSING  {fold_name.upper()}")
        print("=" * 60)

        info = process_fold(fold_name, train_path, test_path, BASE_OUTPUT_DIR)
        all_model_info.append(info)

    # ── write aggregated cross-fold summary ───────────────────────────────────
    save_cross_fold_summary(all_model_info, BASE_OUTPUT_DIR)

    # ── final console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL FOLDS COMPLETE")
    print(f"  Root output dir : {BASE_OUTPUT_DIR}/")
    print("-" * 60)
    print(f"  {'Fold':<10} {'Actions':>8} {'Train seqs':>12} {'1st-order trans':>17}")
    print("-" * 60)
    for info in all_model_info:
        print(
            f"  {info['fold']:<10} "
            f"{info['num_actions']:>8} "
            f"{info['training_sequences']:>12,} "
            f"{info['first_order']['total_transitions']:>17,}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()