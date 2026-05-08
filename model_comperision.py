import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ──────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────

def load_summary(path: str = "outputs_after_model_evaluation/summary.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_fold_metrics(summary: dict) -> tuple[dict, dict]:
    """
    Accepts four summary shapes:

    Shape A – flat list of folds, each fold has 'first_order' / 'second_order' keys:
        [{"fold": 1, "first_order": {...}, "second_order": {...}}, ...]

    Shape B – top-level keys are model names, values are lists of per-fold dicts:
        {"first_order": [{...}, ...], "second_order": [{...}, ...]}

    Shape C – dict with a 'folds' key containing a list of per-fold dicts:
        {"folds": [{"first_order": {...}, "second_order": {...}}, ...]}

    Shape D – aggregated summary, each model key maps to {metric: {"mean": ..., "std": ...}}:
        {"first_order": {"log_likelihood": {"mean": ..., "std": ...}, ...}, "second_order": {...}}
    """
    metrics = ["log_likelihood", "perplexity", "kl_divergence", "unseen_ratio"]

    def _empty():
        return {m: [] for m in metrics}

    first = _empty()
    second = _empty()

    # Shape D – aggregated: metric values are dicts with "mean"/"std" keys
    fo_data = summary.get("first_order", {})
    if (
        isinstance(fo_data, dict)
        and any(m in fo_data for m in metrics)
        and isinstance(next((fo_data[m] for m in metrics if m in fo_data), None), dict)
    ):
        for m in metrics:
            if m in fo_data:
                first[m] = [fo_data[m]["mean"]]
            so_data = summary.get("second_order", {})
            if m in so_data:
                second[m] = [so_data[m]["mean"]]
        return first, second

    # Shape B – top-level model keys, values are lists of per-fold dicts
    if "first_order" in summary and "second_order" in summary:
        for fold in summary["first_order"]:
            for m in metrics:
                if m in fold:
                    first[m].append(fold[m])
        for fold in summary["second_order"]:
            for m in metrics:
                if m in fold:
                    second[m].append(fold[m])
        return first, second

    # Shape A / Shape C – list of folds (top-level or under "folds" key)
    if isinstance(summary, list):
        folds = summary
    elif "folds" in summary:
        folds = summary["folds"]
    else:
        raise ValueError(
            "Unrecognised summary.json structure. "
            "Expected aggregated model keys, top-level model lists, or a list of folds."
        )

    for fold in folds:
        fo = fold.get("first_order", fold.get("first", {}))
        so = fold.get("second_order", fold.get("second", {}))
        for m in metrics:
            if m in fo:
                first[m].append(fo[m])
            if m in so:
                second[m].append(so[m])

    return first, second


# ──────────────────────────────────────────────
# 2. STABILITY
# ──────────────────────────────────────────────

def compute_stability(first: dict, second: dict) -> dict:
    metrics = ["log_likelihood", "perplexity", "kl_divergence", "unseen_ratio"]

    def stats(data: dict) -> dict:
        return {
            m: {
                "mean": float(np.mean(data[m])) if data[m] else None,
                "std":  float(np.std(data[m],  ddof=1)) if len(data[m]) > 1 else None,
            }
            for m in metrics
        }

    fo_stats = stats(first)
    so_stats = stats(second)

    # Count metrics where first-order has lower std
    fo_wins = 0
    so_wins = 0
    for m in metrics:
        fo_std = fo_stats[m]["std"]
        so_std = so_stats[m]["std"]
        if fo_std is not None and so_std is not None:
            if fo_std < so_std:
                fo_wins += 1
            elif so_std < fo_std:
                so_wins += 1

    if fo_wins >= so_wins:
        comparison = (
            f"First-order model is more stable across folds "
            f"(lower std on {fo_wins}/{fo_wins+so_wins} compared metrics). "
            "Second-order models often exhibit higher variance because the larger "
            "transition space is harder to estimate reliably from limited data."
        )
    else:
        comparison = (
            f"Second-order model is more stable across folds "
            f"(lower std on {so_wins}/{fo_wins+so_wins} compared metrics). "
            "This may indicate the dataset is large enough to support the richer "
            "second-order transition space without introducing excess variance."
        )

    return {
        "first_order":  fo_stats,
        "second_order": so_stats,
        "comparison":   comparison,
    }


# ──────────────────────────────────────────────
# 3. ACCURACY
# ──────────────────────────────────────────────

def compute_accuracy(first: dict, second: dict) -> dict:
    metrics = ["log_likelihood", "perplexity", "kl_divergence"]

    def means(data: dict) -> dict:
        return {m: float(np.mean(data[m])) if data[m] else None for m in metrics}

    fo_means = means(first)
    so_means = means(second)

    # Score: higher LL → better; lower perplexity → better; lower KL → better
    fo_score = 0
    so_score = 0
    reasons = []

    ll_fo = fo_means.get("log_likelihood")
    ll_so = so_means.get("log_likelihood")
    if ll_fo is not None and ll_so is not None:
        if ll_fo > ll_so:
            fo_score += 1
            reasons.append(f"higher mean log-likelihood ({ll_fo:.4f} vs {ll_so:.4f})")
        else:
            so_score += 1
            reasons.append(f"higher mean log-likelihood ({ll_so:.4f} vs {ll_fo:.4f})")

    pp_fo = fo_means.get("perplexity")
    pp_so = so_means.get("perplexity")
    if pp_fo is not None and pp_so is not None:
        if pp_fo < pp_so:
            fo_score += 1
            reasons.append(f"lower mean perplexity ({pp_fo:.4f} vs {pp_so:.4f})")
        else:
            so_score += 1
            reasons.append(f"lower mean perplexity ({pp_so:.4f} vs {pp_fo:.4f})")

    kl_fo = fo_means.get("kl_divergence")
    kl_so = so_means.get("kl_divergence")
    if kl_fo is not None and kl_so is not None:
        if kl_fo < kl_so:
            fo_score += 1
            reasons.append(f"lower mean KL divergence ({kl_fo:.4f} vs {kl_so:.4f})")
        else:
            so_score += 1
            reasons.append(f"lower mean KL divergence ({kl_so:.4f} vs {kl_fo:.4f})")

    better = "first_order" if fo_score >= so_score else "second_order"
    reason_str = "; ".join(reasons) if reasons else "metrics are tied"

    return {
        "metrics_mean": {
            "first_order":  fo_means,
            "second_order": so_means,
        },
        "better_model": better,
        "reason": f"{better.replace('_', '-')} model wins on: {reason_str}.",
    }


# ──────────────────────────────────────────────
# 4. SPARSITY TRADE-OFF
# ──────────────────────────────────────────────

def compute_sparsity_tradeoff(first: dict, second: dict) -> dict:
    fo_unseen = float(np.mean(first["unseen_ratio"])) if first["unseen_ratio"] else None
    so_unseen = float(np.mean(second["unseen_ratio"])) if second["unseen_ratio"] else None

    if fo_unseen is not None and so_unseen is not None:
        worse = "second_order" if so_unseen > fo_unseen else "first_order"
        interpretation = (
            f"The second-order model has a mean unseen-transition ratio of {so_unseen:.4f} "
            f"vs {fo_unseen:.4f} for the first-order model. "
            "This is expected: the second-order state space grows quadratically with "
            "vocabulary size, so many (context_t-1, context_t) → next pairs are never "
            "observed in training, leading to zero-probability transitions and higher "
            "perplexity on held-out data. "
            "Without smoothing, this sparsity creates an overfitting risk: the model "
            "memorises training bigrams but cannot generalise to novel continuations."
        )
        conclusion = (
            "first_order handles sparsity better"
            if fo_unseen <= so_unseen
            else "second_order handles sparsity better (unexpectedly low unseen ratio)"
        )
    else:
        interpretation = "unseen_ratio data unavailable for one or both models."
        conclusion = "insufficient data to draw a sparsity conclusion"
        worse = "unknown"

    return {
        "first_order_unseen":  fo_unseen,
        "second_order_unseen": so_unseen,
        "model_with_higher_sparsity": worse,
        "interpretation": interpretation,
        "conclusion": conclusion,
    }


# ──────────────────────────────────────────────
# 5. FINAL RECOMMENDATION
# ──────────────────────────────────────────────

def final_recommendation(stability: dict, accuracy: dict, sparsity: dict) -> dict:
    votes = {"first_order": 0, "second_order": 0}

    # Stability vote
    fo_std_sum = sum(
        v["std"] for v in stability["first_order"].values() if v["std"] is not None
    )
    so_std_sum = sum(
        v["std"] for v in stability["second_order"].values() if v["std"] is not None
    )
    votes["first_order" if fo_std_sum <= so_std_sum else "second_order"] += 1

    # Accuracy vote
    votes[accuracy["better_model"]] += 1

    # Sparsity vote
    conclusion = sparsity["conclusion"]
    if "first_order" in conclusion:
        votes["first_order"] += 1
    elif "second_order" in conclusion and "better" in conclusion:
        votes["second_order"] += 1

    best = max(votes, key=votes.get)

    justification = (
        f"Across all three dimensions the {best.replace('_', '-')} model is preferred. "
        f"Stability: {'first-order' if fo_std_sum <= so_std_sum else 'second-order'} "
        f"model shows lower aggregate variance (sum of stds: "
        f"1st={fo_std_sum:.4f}, 2nd={so_std_sum:.4f}). "
        f"Accuracy: {accuracy['reason']} "
        f"Sparsity: {sparsity['conclusion']}. "
        "Second-order models capture longer dependencies but require substantially more "
        "training data to fill their larger transition matrix; without smoothing they "
        "are prone to sparsity-induced overfitting."
    )

    return {"best_model": best, "justification": justification}


# ──────────────────────────────────────────────
# 6. VISUALISATION
# ──────────────────────────────────────────────

def generate_plots(first: dict, second: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    def safe_std(lst):
        return float(np.std(lst, ddof=1)) if len(lst) > 1 else 0.0

    labels = ["First-Order", "Second-Order"]

    # ── 1. Mean log-likelihood bar chart ──────────────────────────
    fig, ax = plt.subplots()
    vals = [safe_mean(first["log_likelihood"]), safe_mean(second["log_likelihood"])]
    ax.bar(labels, vals)
    ax.set_title("Mean Log-Likelihood Comparison")
    ax.set_ylabel("Mean Log-Likelihood")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "log_likelihood.png"), dpi=150)
    plt.close(fig)

    # ── 2. Mean perplexity bar chart ──────────────────────────────
    fig, ax = plt.subplots()
    vals = [safe_mean(first["perplexity"]), safe_mean(second["perplexity"])]
    ax.bar(labels, vals)
    ax.set_title("Mean Perplexity Comparison")
    ax.set_ylabel("Mean Perplexity")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "perplexity.png"), dpi=150)
    plt.close(fig)

    # ── 3. KL divergence bar chart ────────────────────────────────
    fig, ax = plt.subplots()
    vals = [safe_mean(first["kl_divergence"]), safe_mean(second["kl_divergence"])]
    ax.bar(labels, vals)
    ax.set_title("Mean KL Divergence Comparison")
    ax.set_ylabel("Mean KL Divergence")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kl_divergence.png"), dpi=150)
    plt.close(fig)

    # ── 4. Stability: error-bar plot for all metrics ──────────────
    metrics = ["log_likelihood", "perplexity", "kl_divergence", "unseen_ratio"]
    metric_labels = ["Log-Likelihood", "Perplexity", "KL Divergence", "Unseen Ratio"]

    fo_means = [safe_mean(first[m]) for m in metrics]
    fo_stds  = [safe_std(first[m])  for m in metrics]
    so_means = [safe_mean(second[m]) for m in metrics]
    so_stds  = [safe_std(second[m])  for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, fo_means, width, yerr=fo_stds, capsize=5, label="First-Order")
    ax.bar(x + width / 2, so_means, width, yerr=so_stds, capsize=5, label="Second-Order")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_title("Stability: Mean ± Std Across Folds")
    ax.set_ylabel("Value")
    ax.set_xlabel("Metric")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stability.png"), dpi=150)
    plt.close(fig)

    print(f"[plots] saved to '{out_dir}/'")


# ──────────────────────────────────────────────
# 7. SAVE RESULTS
# ──────────────────────────────────────────────

def save_results(results: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "comparison_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] saved to '{path}'")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    summary_path = "outputs_after_model_evaluation/summary.json"
    out_dir = "model_comparison"

    print("[1/6] Loading summary …")
    summary = load_summary(summary_path)
    first, second = extract_fold_metrics(summary)

    print("[2/6] Computing stability …")
    stability = compute_stability(first, second)

    print("[3/6] Computing accuracy …")
    accuracy = compute_accuracy(first, second)

    print("[4/6] Computing sparsity trade-off …")
    sparsity = compute_sparsity_tradeoff(first, second)

    print("[5/6] Generating final recommendation …")
    recommendation = final_recommendation(stability, accuracy, sparsity)

    results = {
        "stability":           stability,
        "accuracy":            accuracy,
        "sparsity_tradeoff":   sparsity,
        "final_recommendation": recommendation,
    }

    print("[6/6] Generating plots and saving results …")
    generate_plots(first, second, out_dir)
    save_results(results, out_dir)

    print("\n=== Final Recommendation ===")
    print(f"  Best model : {recommendation['best_model']}")
    print(f"  Reasoning  : {recommendation['justification']}")
    print("\nDone.")


if __name__ == "__main__":
    main()