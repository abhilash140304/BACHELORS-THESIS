"""
synthetic_session_generator.py
================================
Generate synthetic user sessions from trained first-order and second-order
Markov models.

Usage:
    python synthetic_session_generator.py

Inputs (set in __main__ or import as module):
    - transition_matrix   : np.ndarray, shape (k, k)   — P(a_t+1 | a_t)
    - transition_tensor   : np.ndarray, shape (k, k, k) — P(a_t+1 | a_t-1, a_t)
    - action_to_index     : dict[str, int]
    - index_to_action     : dict[int, str]
    - real_sessions       : list[list[int]]  — original training sequences (for distributions)

Outputs:
    - synthetic_sessions.json   — generated sessions + metadata
    - (optional) comparison plots if matplotlib is available
"""

import json
import numpy as np
from collections import Counter
from typing import Optional

# ─────────────────────────────────────────────
# 1. EMPIRICAL DISTRIBUTION HELPERS
# ─────────────────────────────────────────────

def compute_start_distribution(real_sessions: list[list[int]], k: int) -> np.ndarray:
    """
    Compute the empirical distribution over starting actions.

    Parameters
    ----------
    real_sessions : list of sessions (each session is a list of action indices)
    k             : vocabulary size (number of unique actions)

    Returns
    -------
    np.ndarray of shape (k,) — probability of each action being a session start
    """
    counts = np.zeros(k)
    for session in real_sessions:
        if len(session) > 0:
            counts[session[0]] += 1
    total = counts.sum()
    if total == 0:
        return np.ones(k) / k          # uniform fallback
    return counts / total


def compute_start_pair_distribution(real_sessions: list[list[int]], k: int) -> np.ndarray:
    """
    Compute the empirical joint distribution over starting (a1, a2) pairs.

    Parameters
    ----------
    real_sessions : list of sessions
    k             : vocabulary size

    Returns
    -------
    np.ndarray of shape (k, k) — probability of each (a1, a2) pair starting a session
    """
    counts = np.zeros((k, k))
    for session in real_sessions:
        if len(session) >= 2:
            counts[session[0], session[1]] += 1
    total = counts.sum()
    if total == 0:
        return np.ones((k, k)) / (k * k)   # uniform fallback
    return counts / total


def compute_session_length_distribution(real_sessions: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical distribution over session lengths.

    Returns
    -------
    lengths : unique session lengths observed
    probs   : corresponding probabilities
    """
    lengths = [len(s) for s in real_sessions if len(s) > 0]
    counter = Counter(lengths)
    unique_lengths = np.array(sorted(counter.keys()))
    counts = np.array([counter[l] for l in unique_lengths], dtype=float)
    probs = counts / counts.sum()
    return unique_lengths, probs


# ─────────────────────────────────────────────
# 2. FIRST-ORDER SESSION SAMPLER
# ─────────────────────────────────────────────

def sample_first_order_session(
    transition_matrix: np.ndarray,
    start_dist: np.ndarray,
    target_length: int,
    terminal_state: Optional[int] = None,
) -> list[int]:
    """
    Generate a single synthetic session using a first-order Markov model.

    Parameters
    ----------
    transition_matrix : np.ndarray, shape (k, k)
        Row-stochastic matrix. transition_matrix[i, j] = P(a_t+1=j | a_t=i).
    start_dist        : np.ndarray, shape (k,)
        Probability distribution over initial states.
    target_length     : int
        Maximum (or target) number of actions in the session.
    terminal_state    : int or None
        If set, sampling stops early when this action is drawn.

    Returns
    -------
    list[int] — sequence of action indices
    """
    k = len(start_dist)

    # Sample starting state
    current = int(np.random.choice(k, p=start_dist))
    session = [current]

    for _ in range(target_length - 1):
        row = transition_matrix[current]

        # Safety: re-normalise in case of floating-point drift
        row_sum = row.sum()
        if row_sum == 0:
            # Degenerate row — sample uniformly
            row = np.ones(k) / k
        else:
            row = row / row_sum

        next_action = int(np.random.choice(k, p=row))
        session.append(next_action)

        # Stop if we hit an absorbing / terminal state
        if terminal_state is not None and next_action == terminal_state:
            break

        current = next_action

    return session


# ─────────────────────────────────────────────
# 3. SECOND-ORDER SESSION SAMPLER
# ─────────────────────────────────────────────

def sample_second_order_session(
    transition_tensor: np.ndarray,
    transition_matrix: np.ndarray,
    start_pair_dist: np.ndarray,
    target_length: int,
    terminal_state: Optional[int] = None,
) -> list[int]:
    """
    Generate a single synthetic session using a second-order Markov model.

    Falls back to first-order when a (prev, curr) context pair was not
    observed during training (i.e., the corresponding row sums to zero).

    Parameters
    ----------
    transition_tensor  : np.ndarray, shape (k, k, k)
        transition_tensor[i, j, l] = P(a_t+1=l | a_t-1=i, a_t=j).
    transition_matrix  : np.ndarray, shape (k, k)
        First-order fallback matrix.
    start_pair_dist    : np.ndarray, shape (k, k)
        Joint distribution over starting (a1, a2) pairs.
    target_length      : int
        Maximum number of actions in the session.
    terminal_state     : int or None
        Early-stopping absorbing state (optional).

    Returns
    -------
    list[int] — sequence of action indices
    """
    k = transition_tensor.shape[0]

    # ── Sample the initial pair (a1, a2) ────────────────────────────────
    flat_probs = start_pair_dist.ravel()
    flat_probs = flat_probs / flat_probs.sum()          # ensure sum-to-1
    flat_idx   = int(np.random.choice(k * k, p=flat_probs))
    prev  = flat_idx // k
    curr  = flat_idx  % k

    session = [prev, curr]

    for _ in range(target_length - 2):
        # ── Try second-order transition ──────────────────────────────────
        row = transition_tensor[prev, curr]
        row_sum = row.sum()

        if row_sum == 0:
            # ── Fallback: use first-order transition ─────────────────────
            row = transition_matrix[curr]
            row_sum = row.sum()

            if row_sum == 0:
                # Last resort: sample uniformly
                row = np.ones(k) / k
            else:
                row = row / row_sum
        else:
            row = row / row_sum

        next_action = int(np.random.choice(k, p=row))
        session.append(next_action)

        if terminal_state is not None and next_action == terminal_state:
            break

        # Slide the context window forward
        prev = curr
        curr = next_action

    return session


# ─────────────────────────────────────────────
# 4. BATCH SESSION GENERATOR
# ─────────────────────────────────────────────

def generate_sessions(
    transition_matrix: np.ndarray,
    transition_tensor: np.ndarray,
    real_sessions: list[list[int]],
    n_sessions: int = 500,
    max_length: int = 50,
    use_empirical_lengths: bool = True,
    terminal_state: Optional[int] = None,
    random_seed: int = 42,
) -> dict:
    """
    Generate N synthetic sessions using both first- and second-order models.

    Parameters
    ----------
    transition_matrix      : shape (k, k)
    transition_tensor      : shape (k, k, k)
    real_sessions          : original training sessions (list of int lists)
    n_sessions             : number of sessions to generate per model
    max_length             : fallback / hard cap on session length
    use_empirical_lengths  : if True, sample per-session length from real data
    terminal_state         : optional absorbing state index
    random_seed            : for full reproducibility

    Returns
    -------
    dict with keys:
        first_order_sessions  : list[list[int]]
        second_order_sessions : list[list[int]]
        metadata              : dict
    """
    np.random.seed(random_seed)

    k = transition_matrix.shape[0]

    # ── Precompute empirical distributions ──────────────────────────────
    start_dist      = compute_start_distribution(real_sessions, k)
    start_pair_dist = compute_start_pair_distribution(real_sessions, k)

    if use_empirical_lengths and len(real_sessions) > 0:
        lengths, length_probs = compute_session_length_distribution(real_sessions)
        # Pre-draw all target lengths at once for efficiency
        sampled_lengths = np.random.choice(lengths, size=n_sessions, p=length_probs)
        sampled_lengths = np.clip(sampled_lengths, 2, max_length)   # at least 2 for 2nd-order
    else:
        sampled_lengths = np.full(n_sessions, max_length)

    # ── Generate sessions ────────────────────────────────────────────────
    first_order_sessions  = []
    second_order_sessions = []

    for i in range(n_sessions):
        target_len = int(sampled_lengths[i])

        # First-order
        fo_session = sample_first_order_session(
            transition_matrix=transition_matrix,
            start_dist=start_dist,
            target_length=target_len,
            terminal_state=terminal_state,
        )
        first_order_sessions.append(fo_session)

        # Second-order
        so_session = sample_second_order_session(
            transition_tensor=transition_tensor,
            transition_matrix=transition_matrix,
            start_pair_dist=start_pair_dist,
            target_length=target_len,
            terminal_state=terminal_state,
        )
        second_order_sessions.append(so_session)

    result = {
        "first_order_sessions":  first_order_sessions,
        "second_order_sessions": second_order_sessions,
        "metadata": {
            "num_sessions":      n_sessions,
            "max_length":        max_length,
            "use_empirical_lengths": use_empirical_lengths,
            "generation_type":   "probabilistic_sampling",
            "random_seed":       random_seed,
            "vocab_size":        k,
            "terminal_state":    terminal_state,
        },
    }
    return result


# ─────────────────────────────────────────────
# 5. OUTPUT: SAVE TO JSON
# ─────────────────────────────────────────────

def save_sessions(sessions_dict: dict, filepath: str = "synthetic_sessions.json") -> None:
    """Serialise the generated sessions dict to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sessions_dict, f, indent=2)
    print(f"[✓] Saved synthetic sessions → {filepath}")


# ─────────────────────────────────────────────
# 6. BONUS: COMPARISON PLOTS (real vs synthetic)
# ─────────────────────────────────────────────

def compare_distributions(
    real_sessions: list[list[int]],
    synthetic_fo: list[list[int]],
    synthetic_so: list[list[int]],
    k: int,
    save_path: Optional[str] = "comparison.png",
) -> None:
    """
    Plot real vs synthetic distributions:
      (a) Session length distribution
      (b) Action frequency distribution

    Requires matplotlib. Silently skips if not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not found — skipping comparison plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Real vs Synthetic Sessions", fontsize=14, fontweight="bold")

    # ── (a) Session length distributions ────────────────────────────────
    ax = axes[0]
    real_lengths = [len(s) for s in real_sessions]
    fo_lengths   = [len(s) for s in synthetic_fo]
    so_lengths   = [len(s) for s in synthetic_so]

    max_len = max(max(real_lengths, default=0),
                  max(fo_lengths,   default=0),
                  max(so_lengths,   default=0))
    bins = np.arange(1, max_len + 2) - 0.5

    ax.hist(real_lengths, bins=bins, density=True, alpha=0.55, label="Real",
            color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.hist(fo_lengths,   bins=bins, density=True, alpha=0.55, label="1st-order",
            color="#FF5722", edgecolor="white", linewidth=0.5)
    ax.hist(so_lengths,   bins=bins, density=True, alpha=0.55, label="2nd-order",
            color="#4CAF50", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Session Length")
    ax.set_ylabel("Density")
    ax.set_title("Session Length Distribution")
    ax.legend()

    # ── (b) Action frequency distributions ──────────────────────────────
    ax = axes[1]

    def action_freq(sessions, vocab_size):
        counts = np.zeros(vocab_size)
        for s in sessions:
            for a in s:
                counts[a] += 1
        total = counts.sum()
        return counts / total if total > 0 else counts

    real_freq = action_freq(real_sessions, k)
    fo_freq   = action_freq(synthetic_fo,  k)
    so_freq   = action_freq(synthetic_so,  k)

    x = np.arange(k)
    width = 0.28
    ax.bar(x - width, real_freq, width, label="Real",       color="#2196F3", alpha=0.8)
    ax.bar(x,         fo_freq,   width, label="1st-order",  color="#FF5722", alpha=0.8)
    ax.bar(x + width, so_freq,   width, label="2nd-order",  color="#4CAF50", alpha=0.8)

    ax.set_xlabel("Action Index")
    ax.set_ylabel("Relative Frequency")
    ax.set_title("Action Frequency Distribution")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Saved comparison plot → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 7. MAIN — minimal demo with synthetic data
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Replace these with your real trained objects ─────────────────────
    # Example: loading from disk
    #
    #   import numpy as np, json
    #   transition_matrix = np.load("transition_matrix.npy")
    #   transition_tensor = np.load("transition_tensor.npy")
    #   with open("action_to_index.json") as f:
    #       action_to_index = json.load(f)
    #   index_to_action = {v: k for k, v in action_to_index.items()}
    #   with open("real_sessions.json") as f:
    #       real_sessions = json.load(f)
    #
    # ── Demo: build toy objects so the script is self-contained ──────────

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    k = 8   # vocabulary size

    # Toy Laplace-smoothed transition matrix  (k × k)
    raw_tm = np.random.randint(1, 20, size=(k, k)).astype(float)
    transition_matrix = raw_tm / raw_tm.sum(axis=1, keepdims=True)

    # Toy second-order tensor  (k × k × k)
    raw_tt = np.random.randint(1, 10, size=(k, k, k)).astype(float)
    # Normalise so tensor[i, j, :] is a valid distribution
    row_sums = raw_tt.sum(axis=2, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid division by zero
    transition_tensor = raw_tt / row_sums

    action_to_index = {f"action_{i}": i for i in range(k)}
    index_to_action = {i: f"action_{i}" for i in range(k)}

    # Toy real sessions (lengths 3–12)
    real_sessions = [
        list(np.random.randint(0, k, size=np.random.randint(3, 13)))
        for _ in range(300)
    ]

    # ── Generate synthetic sessions ───────────────────────────────────────
    print("Generating synthetic sessions …")
    results = generate_sessions(
        transition_matrix     = transition_matrix,
        transition_tensor     = transition_tensor,
        real_sessions         = real_sessions,
        n_sessions            = 500,
        max_length            = 50,
        use_empirical_lengths = True,   # sample lengths from real data
        terminal_state        = None,   # set to an int if you have a terminal action
        random_seed           = RANDOM_SEED,
    )

    print(f"  First-order  sessions generated : {len(results['first_order_sessions'])}")
    print(f"  Second-order sessions generated : {len(results['second_order_sessions'])}")
    print(f"  Example 1st-order session       : {results['first_order_sessions'][0]}")
    print(f"  Example 2nd-order session       : {results['second_order_sessions'][0]}")

    # ── Save JSON output ─────────────────────────────────────────────────
    save_sessions(results, filepath="synthetic_sessions.json")

    # ── (Bonus) comparison plots ─────────────────────────────────────────
    compare_distributions(
        real_sessions  = real_sessions,
        synthetic_fo   = results["first_order_sessions"],
        synthetic_so   = results["second_order_sessions"],
        k              = k,
        save_path      = "comparison.png",
    )