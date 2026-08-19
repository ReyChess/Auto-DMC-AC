#!/usr/bin/env python3
from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "reproducible_baselines" / "external_baseline_folds.csv"
AUTO_ROOT = ROOT / "results_predictor_ablation"
OUT = ROOT / "external_baseline_analysis"
OUT.mkdir(exist_ok=True)

MODELS = ["Auto-DMC-AC", "CART", "Logistic", "BernoulliNB", "RandomForest"]
BASELINES = MODELS[1:]


def holm(pairs):
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    adjusted = [0.0] * len(pairs)
    running = 0.0
    for k, idx in enumerate(order):
        running = max(running, (len(pairs) - k) * pairs[idx][1])
        adjusted[idx] = min(1.0, running)
    return adjusted


ext = pd.read_csv(EXT)
if len(ext) != 600:
    raise RuntimeError(f"Expected 600 external rows (15x10x4), got {len(ext)}")

ext_ds = (
    ext.groupby(["dataset", "model"], as_index=False)
    [["outer_accuracy", "outer_macro_f1"]]
    .mean()
)

auto_rows = []
for path in glob.glob(str(AUTO_ROOT / "*" / "predictor_ablation.csv")):
    dataset = os.path.basename(os.path.dirname(path))
    z = pd.read_csv(path)
    z = z[z["ablation"] == "full"]
    if len(z) != 10:
        raise RuntimeError(f"Expected 10 full folds for {dataset}, got {len(z)}")
    auto_rows.append(
        {"dataset": dataset, "accuracy": z.accuracy.mean(), "macro_f1": z.macro_f1.mean()}
    )
auto = pd.DataFrame(auto_rows).sort_values("dataset").reset_index(drop=True)
if len(auto) != 15:
    raise RuntimeError(f"Expected 15 Auto-DMC-AC datasets, got {len(auto)}")

wide = auto.copy()
wide = wide.rename(columns={"accuracy": "Auto-DMC-AC_accuracy", "macro_f1": "Auto-DMC-AC_macro_f1"})
for model in BASELINES:
    q = ext_ds[ext_ds.model == model].set_index("dataset").reindex(auto.dataset)
    wide[f"{model}_accuracy"] = q.outer_accuracy.to_numpy()
    wide[f"{model}_macro_f1"] = q.outer_macro_f1.to_numpy()
wide.to_csv(OUT / "per_dataset_comparison.csv", index=False)

summary_rows = []
tests_rows = []
rank_rows = []
for metric in ("accuracy", "macro_f1"):
    vals = [wide[f"{m}_{metric}"].to_numpy() for m in MODELS]
    stat, fp = friedmanchisquare(*vals)
    ranks = np.vstack([
        rankdata(-np.asarray([v[i] for v in vals]), method="average")
        for i in range(len(wide))
    ])
    for m, v, rank in zip(MODELS, vals, ranks.mean(axis=0)):
        summary_rows.append({"metric": metric, "model": m, "mean": v.mean(), "average_rank": rank})
        rank_rows.append({"metric": metric, "model": m, "average_rank": rank, "friedman_stat": stat, "friedman_p": fp})

    auto_v = vals[0]
    raw = []
    pending = []
    for model, base_v in zip(BASELINES, vals[1:]):
        diff = auto_v - base_v
        p = float(wilcoxon(diff, alternative="two-sided").pvalue)
        raw.append((model, p))
        pending.append((model, base_v, diff, p))
    adjusted = holm(raw)
    for (model, base_v, diff, p), hp in zip(pending, adjusted):
        tests_rows.append({
            "metric": metric,
            "baseline": model,
            "auto_mean": auto_v.mean(),
            "baseline_mean": base_v.mean(),
            "auto_minus_baseline": diff.mean(),
            "auto_wins": int(np.sum(diff > 1e-12)),
            "ties": int(np.sum(np.abs(diff) <= 1e-12)),
            "auto_losses": int(np.sum(diff < -1e-12)),
            "wilcoxon_p": p,
            "holm_p": hp,
        })

summary = pd.DataFrame(summary_rows)
tests = pd.DataFrame(tests_rows)
ranks_df = pd.DataFrame(rank_rows)
summary.to_csv(OUT / "overall_summary.csv", index=False)
tests.to_csv(OUT / "paired_tests.csv", index=False)
ranks_df.to_csv(OUT / "friedman_ranks.csv", index=False)

cart = ext[ext.model == "CART"].copy()
cart[["tree_depth", "tree_leaves", "tree_nodes"]] = cart[["tree_depth", "tree_leaves", "tree_nodes"]].apply(pd.to_numeric)
cart_complexity = cart.groupby("dataset", as_index=False)[["tree_depth", "tree_leaves", "tree_nodes"]].mean()
cart_complexity.to_csv(OUT / "cart_complexity.csv", index=False)

with (OUT / "README.md").open("w") as f:
    f.write("# Reproducible external-baseline comparison\n\n")
    f.write("Auto-DMC-AC is compared against four classifiers run on the exact same 10 outer folds for 15 datasets. ")
    f.write("External models use binary item indicators, with the vocabulary fitted exclusively on the corresponding training data. ")
    f.write("Hyperparameters are selected using a 3-fold stratified split of the outer-training data only, maximizing macro-F1 and breaking ties by accuracy and then fixed candidate order. ")
    f.write("The outer test set is never used for model or hyperparameter selection.\n\n")
    f.write("The inferential unit is the dataset (n=15). Pairwise comparisons use two-sided Wilcoxon signed-rank tests on the 15 dataset means, ")
    f.write("with Holm correction across the four Auto-DMC-AC-vs-baseline tests separately for each metric. A Friedman test and average ranks are also reported across all five methods.\n\n")
    f.write("## Overall means and ranks\n\n")
    f.write("```text\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n```\n\n")
    f.write("## Paired tests\n\n")
    f.write("```text\n" + tests.to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n```\n\n")
    for metric in ("accuracy", "macro_f1"):
        rr = ranks_df[ranks_df.metric == metric].iloc[0]
        f.write(f"Friedman {metric}: chi-square={rr.friedman_stat:.6f}, p={rr.friedman_p:.6f}.\n\n")
    f.write("Interpretation: Random Forest is the only external baseline significantly better than Auto-DMC-AC in accuracy after Holm correction. ")
    f.write("No pairwise macro-F1 difference remains significant after Holm correction. The result supports competitive predictive performance relative to transparent/simple baselines, ")
    f.write("while documenting a measurable accuracy trade-off relative to the stronger non-symbolic Random Forest reference.\n")

print(summary.to_string(index=False))
print()
print(tests.to_string(index=False))
print()
for metric in ("accuracy", "macro_f1"):
    rr = ranks_df[ranks_df.metric == metric].iloc[0]
    print(f"Friedman {metric}: stat={rr.friedman_stat:.6f}, p={rr.friedman_p:.6g}")
