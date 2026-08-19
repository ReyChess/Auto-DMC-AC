#!/usr/bin/env python3
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
PARTITIONS = ROOT / "partitions_work" / "Partitions"
OUT = ROOT / "reproducible_baselines"
OUT.mkdir(exist_ok=True)
SEED = 20260715

DATASETS = [
    "Anneal", "Breast", "Flare", "Glass", "Heart", "Hepatitis",
    "HorseColic", "Ionosphere", "Iris", "Led7", "LetRecog",
    "Mushroom", "PageBlocks", "PenDigits", "Pima",
]


def read_dat(path):
    items, labels = [], []
    with path.open() as f:
        for line in f:
            vals = [int(z) for z in line.split()]
            if not vals:
                continue
            items.append(vals[:-1])
            labels.append(vals[-1])
    return items, np.asarray(labels, dtype=int)


def vectorize(train_items, test_items):
    vocab = sorted({item for row in train_items for item in row})
    pos = {item: j for j, item in enumerate(vocab)}

    def make(rows):
        rr, cc = [], []
        for i, row in enumerate(rows):
            for item in row:
                j = pos.get(item)
                if j is not None:
                    rr.append(i)
                    cc.append(j)
        data = np.ones(len(rr), dtype=np.float64)
        return csr_matrix((data, (rr, cc)), shape=(len(rows), len(vocab)))

    return make(train_items), make(test_items), len(vocab)


def model_specs(seed):
    return {
        "CART": (
            DecisionTreeClassifier(random_state=seed),
            [
                {"max_depth": depth, "min_samples_leaf": leaf, "ccp_alpha": alpha}
                for depth in (3, 5, 10, None)
                for leaf in (10, 5, 1)
                for alpha in (0.0, 0.001)
            ],
        ),
        "Logistic": (
            LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed),
            [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0)],
        ),
        "BernoulliNB": (
            BernoulliNB(),
            [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0)],
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, max_features="sqrt", n_jobs=-1, random_state=seed),
            [
                {"max_depth": depth, "min_samples_leaf": leaf}
                for depth in (20, None)
                for leaf in (5, 1)
            ],
        ),
    }


fieldnames = [
    "dataset", "outer_fold", "model", "selected_params", "inner_macro_f1",
    "inner_accuracy", "outer_accuracy", "outer_macro_f1", "correct", "total",
    "features", "selection_seconds", "fit_seconds", "prediction_seconds",
    "tree_depth", "tree_leaves", "tree_nodes",
]
rows_out = []

for dataset in DATASETS:
    ds_dir = PARTITIONS / f"{dataset}Partition"
    for outer in range(1, 11):
        train_items, y_train = read_dat(ds_dir / f"Dataset{outer}.dat")
        test_items, y_test = read_dat(ds_dir / f"{outer}.dat")

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + outer)
        cached_inner = []
        for tr_idx, va_idx in skf.split(np.zeros(len(y_train)), y_train):
            tr_items = [train_items[i] for i in tr_idx]
            va_items = [train_items[i] for i in va_idx]
            xtr, xva, _ = vectorize(tr_items, va_items)
            cached_inner.append((xtr, y_train[tr_idx], xva, y_train[va_idx]))

        x_train, x_test, n_features = vectorize(train_items, test_items)

        for model_name, (base_model, candidates) in model_specs(SEED + outer).items():
            tsel = time.perf_counter()
            scored = []
            for order, params in enumerate(candidates):
                f1s, accs = [], []
                for xtr, ytr, xva, yva in cached_inner:
                    model = clone(base_model).set_params(**params)
                    model.fit(xtr, ytr)
                    pred = model.predict(xva)
                    # Match Auto-DMC-AC exactly: macro-F1 is averaged over the
                    # classes known to the corresponding training fold, with a
                    # zero contribution when a known class has no denominator
                    # in validation/test.
                    f1s.append(f1_score(yva, pred, labels=np.unique(ytr), average="macro", zero_division=0))
                    accs.append(accuracy_score(yva, pred))
                scored.append((float(np.mean(f1s)), float(np.mean(accs)), -order, params))
            best_f1, best_acc, _, best_params = max(scored, key=lambda z: (z[0], z[1], z[2]))
            selection_seconds = time.perf_counter() - tsel

            final_model = clone(base_model).set_params(**best_params)
            t0 = time.perf_counter()
            final_model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            pred = final_model.predict(x_test)
            prediction_seconds = time.perf_counter() - t1

            correct = int(np.sum(pred == y_test))
            row = {
                "dataset": dataset,
                "outer_fold": outer,
                "model": model_name,
                "selected_params": json.dumps(best_params, sort_keys=True),
                "inner_macro_f1": best_f1,
                "inner_accuracy": best_acc,
                "outer_accuracy": accuracy_score(y_test, pred),
                "outer_macro_f1": f1_score(y_test, pred, labels=np.unique(y_train), average="macro", zero_division=0),
                "correct": correct,
                "total": len(y_test),
                "features": n_features,
                "selection_seconds": selection_seconds,
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
                "tree_depth": "",
                "tree_leaves": "",
                "tree_nodes": "",
            }
            if model_name == "CART":
                row["tree_depth"] = final_model.get_depth()
                row["tree_leaves"] = final_model.get_n_leaves()
                row["tree_nodes"] = final_model.tree_.node_count
            rows_out.append(row)
            print(dataset, outer, model_name,
                  f"acc={row['outer_accuracy']:.4f}", f"f1={row['outer_macro_f1']:.4f}",
                  best_params, flush=True)

            with (OUT / "external_baseline_folds.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows_out)
