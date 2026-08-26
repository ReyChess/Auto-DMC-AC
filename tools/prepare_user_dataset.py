#!/usr/bin/env python3
"""Prepare user datasets for Auto-DMC-AC.

Subcommands
-----------
outer-cv         Raw CSV/table -> stratified external partitions with fold-fitted preprocessing.
transactional-cv Already-discrete transactional file -> reproducible stratified partitions.
encode           Raw CSV/table -> one transactional file (exploratory/non-CV use).

Numeric methods: lucs-kdd-style, mdlp, equal-width, equal-frequency.

For outer-cv, learned preprocessing parameters are fitted on each outer TRAIN fold
only and then applied unchanged to its TEST fold. Namespace planning only assigns
stable compact integer IDs; it does not use labels to learn cut points or mappings.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

MDLP_CAPACITY_DEFAULT = 256


def entropy(y):
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def candidate_cutpoints(x, y):
    order = np.argsort(x, kind="mergesort")
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    cuts = []
    for i in range(len(xs) - 1):
        if xs[i] != xs[i + 1] and ys[i] != ys[i + 1]:
            cuts.append(float((xs[i] + xs[i + 1]) / 2.0))
    return cuts


def mdlp_accept(x, y, cut):
    left = x <= cut
    right = ~left
    n = len(y)
    n1, n2 = int(left.sum()), int(right.sum())
    if n1 == 0 or n2 == 0:
        return False, 0.0
    h = entropy(y)
    h1, h2 = entropy(y[left]), entropy(y[right])
    gain = h - (n1 / n) * h1 - (n2 / n) * h2
    k = len(np.unique(y))
    k1 = len(np.unique(y[left]))
    k2 = len(np.unique(y[right]))
    delta = math.log2(max(1, 3**k - 2)) - (k * h - k1 * h1 - k2 * h2)
    threshold = (math.log2(max(1, n - 1)) + delta) / n
    return gain > threshold, gain


def mdlp_recursive(x, y):
    if len(x) < 2 or len(np.unique(y)) <= 1:
        return []
    best_cut, best_gain, best_ok = None, -1.0, False
    for cut in candidate_cutpoints(x, y):
        ok, gain = mdlp_accept(x, y, cut)
        if gain > best_gain:
            best_cut, best_gain, best_ok = cut, gain, ok
    if best_cut is None or not best_ok:
        return []
    left = x <= best_cut
    return mdlp_recursive(x[left], y[left]) + [float(best_cut)] + mdlp_recursive(x[~left], y[~left])


def equal_width_edges(values, n_bins):
    x = np.asarray(values, dtype=float)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if mn == mx:
        return [mn, mx]
    return np.linspace(mn, mx, n_bins + 1).tolist()


def equal_frequency_edges(values, n_bins):
    x = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)).astype(float))
    if len(edges) < 2:
        v = float(edges[0])
        return [v, v]
    return edges.tolist()


def cutpoints_to_edges(values, cuts):
    x = np.asarray(values, dtype=float)
    return [float(np.nanmin(x))] + sorted(map(float, cuts)) + [float(np.nanmax(x))]


def assign_bin(v, edges):
    if len(edges) <= 2:
        return 0
    return int(np.digitize([float(v)], np.asarray(edges[1:-1]), right=False)[0])


def parse_bins_spec(spec, numeric_columns, default_bins):
    out = {c: default_bins for c in numeric_columns}
    if spec:
        for token in spec.split(","):
            name, value = token.split(":", 1)
            out[name.strip()] = int(value)
    return out


def numeric_columns_for(df, class_column, explicit):
    if explicit:
        cols = [x.strip() for x in explicit.split(",") if x.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Unknown numeric columns: {missing}")
        return cols
    return [c for c in df.columns if c != class_column and pd.api.types.is_numeric_dtype(df[c])]


class NamespacePlan:
    """Compact, fold-stable item IDs without using test data to fit transformations."""

    def __init__(self, df, class_column, numeric_columns, method, bins_by_column, mdlp_capacity):
        self.class_column = class_column
        self.features = [c for c in df.columns if c != class_column]
        self.numeric_columns = set(numeric_columns)
        self.offsets: Dict[str, int] = {}
        self.capacities: Dict[str, int] = {}
        self.nominal_ids: Dict[str, Dict[str, int]] = {}
        next_id = 1

        for c in self.features:
            self.offsets[c] = next_id
            if c in self.numeric_columns:
                capacity = mdlp_capacity if method == "mdlp" else max(1, int(bins_by_column[c]))
            else:
                vals = df[c].astype("string").fillna("__MISSING__")
                cats = sorted(map(str, vals.unique()))
                capacity = max(1, len(cats))
                self.nominal_ids[c] = {cat: next_id + i for i, cat in enumerate(cats)}
            self.capacities[c] = capacity
            next_id += capacity

        classes = sorted(map(str, df[class_column].astype("string").unique()))
        self.class_items = {label: next_id + i for i, label in enumerate(classes)}
        self.max_item_id = next_id + len(classes) - 1

    def numeric_ids(self, column, n_states):
        capacity = self.capacities[column]
        if n_states > capacity:
            raise ValueError(
                f"{column!r} produced {n_states} numeric states, above the reserved capacity {capacity}. "
                "Increase --mdlp-max-bins if using MDLP."
            )
        start = self.offsets[column]
        return list(range(start, start + n_states))

    def to_metadata(self):
        return {
            "feature_offsets": self.offsets,
            "feature_capacities": self.capacities,
            "class_items": self.class_items,
            "max_item_id": self.max_item_id,
        }


class FoldEncoder:
    def __init__(self, class_column, numeric_columns, method, bins_by_column, namespace):
        self.class_column = class_column
        self.numeric_columns = set(numeric_columns)
        self.method = method
        self.bins_by_column = bins_by_column
        self.namespace = namespace
        self.models = {}

    def _fit_numeric(self, values, y, col):
        vals = pd.to_numeric(values, errors="coerce")
        median = float(vals.median())
        vals = vals.fillna(median).to_numpy(float)
        if self.method in ("lucs-kdd-style", "equal-width"):
            edges = equal_width_edges(vals, self.bins_by_column[col])
        elif self.method == "equal-frequency":
            edges = equal_frequency_edges(vals, self.bins_by_column[col])
        elif self.method == "mdlp":
            edges = cutpoints_to_edges(vals, mdlp_recursive(vals, np.asarray(y)))
        else:
            raise ValueError(self.method)
        n_states = max(1, len(edges) - 1)
        return {
            "type": "numeric",
            "median": median,
            "edges": edges,
            "item_ids": self.namespace.numeric_ids(col, n_states),
        }

    def fit(self, df):
        y = df[self.class_column].astype(str).to_numpy()
        for c in self.namespace.features:
            if c in self.numeric_columns:
                self.models[c] = self._fit_numeric(df[c], y, c)
            else:
                vals = df[c].astype("string").fillna("__MISSING__")
                observed = set(map(str, vals.unique()))
                mapping = {k: v for k, v in self.namespace.nominal_ids[c].items() if k in observed}
                self.models[c] = {"type": "nominal", "mapping": mapping}
        return self

    def transform(self, df):
        rows = []
        for _, row in df.iterrows():
            items = []
            for c, model in self.models.items():
                if model["type"] == "numeric":
                    try:
                        value = float(row[c])
                        if np.isnan(value):
                            value = model["median"]
                    except Exception:
                        value = model["median"]
                    b = max(0, min(assign_bin(value, model["edges"]), len(model["item_ids"]) - 1))
                    items.append(model["item_ids"][b])
                else:
                    key = str(row[c]) if not pd.isna(row[c]) else "__MISSING__"
                    if key in model["mapping"]:
                        items.append(model["mapping"][key])
            cls = str(row[self.class_column])
            items.append(self.namespace.class_items[cls])
            rows.append(items)
        return rows

    def metadata(self):
        return {
            "numeric_method": self.method,
            "bins_by_column": self.bins_by_column,
            "feature_models": self.models,
            "namespace": self.namespace.to_metadata(),
        }


def write_rows(rows, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")


def write_class_files(out, class_items, labels):
    counts = pd.Series(labels, dtype="string").value_counts()
    with (out / "Classes.dat").open("w", encoding="utf-8") as f:
        for label, item_id in class_items.items():
            f.write(f"{item_id} {int(counts[label])}\n")
    with (out / "class_dictionary.csv").open("w", encoding="utf-8") as f:
        f.write("original_label,class_item_id,count\n")
        for label, item_id in class_items.items():
            f.write(f"{label},{item_id},{int(counts[label])}\n")


def encode(args):
    df = pd.read_csv(args.input)
    nums = numeric_columns_for(df, args.class_column, args.numeric_columns)
    bins = parse_bins_spec(args.bins_by_column, nums, args.default_bins)
    ns = NamespacePlan(df, args.class_column, nums, args.numeric_method, bins, args.mdlp_max_bins)
    enc = FoldEncoder(args.class_column, nums, args.numeric_method, bins, ns).fit(df)
    write_rows(enc.transform(df), args.output)
    Path(args.output + ".metadata.json").write_text(json.dumps(enc.metadata(), indent=2), encoding="utf-8")


def outer_cv(args):
    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv(args.input)
    if args.class_column not in df.columns:
        raise ValueError(f"Class column not found: {args.class_column}")
    nums = numeric_columns_for(df, args.class_column, args.numeric_columns)
    bins = parse_bins_spec(args.bins_by_column, nums, args.default_bins)
    ns = NamespacePlan(df, args.class_column, nums, args.numeric_method, bins, args.mdlp_max_bins)
    y = df[args.class_column].astype(str).to_numpy()
    cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_mode": "raw-tabular",
        "numeric_method": args.numeric_method,
        "outer_folds": args.outer_folds,
        "seed": args.seed,
        "namespace": ns.to_metadata(),
        "folds": [],
    }

    for k, (tr, te) in enumerate(cv.split(np.zeros(len(df)), y), 1):
        train = df.iloc[tr].reset_index(drop=True)
        test = df.iloc[te].reset_index(drop=True)
        enc = FoldEncoder(args.class_column, nums, args.numeric_method, bins, ns).fit(train)
        train_rows = enc.transform(train)
        test_rows = enc.transform(test)
        write_rows(train_rows, out / f"Dataset{k}.dat")
        write_rows(test_rows, out / f"{k}.dat")
        (out / f"preprocessing_fold_{k}.json").write_text(
            json.dumps(enc.metadata(), indent=2), encoding="utf-8"
        )
        manifest["folds"].append({"fold": k, "train_rows": len(train_rows), "test_rows": len(test_rows)})

    write_class_files(out, ns.class_items, df[args.class_column].astype(str))
    (out / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_transactional(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = [int(x) for x in text.split()]
            except ValueError as exc:
                raise ValueError(f"Non-integer token at line {line_no}") from exc
            if len(row) < 1:
                raise ValueError(f"Empty transaction at line {line_no}")
            rows.append(row)
    if not rows:
        raise ValueError("No transactions found")
    return rows


def transactional_cv(args):
    from sklearn.model_selection import StratifiedKFold

    rows = read_transactional(args.input)
    y = np.asarray([row[-1] for row in rows])
    cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dummy = np.zeros(len(rows))
    for k, (tr, te) in enumerate(cv.split(dummy, y), 1):
        write_rows([rows[i] for i in tr], out / f"Dataset{k}.dat")
        write_rows([rows[i] for i in te], out / f"{k}.dat")
    classes, counts = np.unique(y, return_counts=True)
    with (out / "Classes.dat").open("w", encoding="utf-8") as f:
        for cls, count in zip(classes, counts):
            f.write(f"{int(cls)} {int(count)}\n")
    manifest = {
        "input_mode": "already-transactional",
        "outer_folds": args.outer_folds,
        "seed": args.seed,
        "rows": len(rows),
        "classes": {str(int(c)): int(n) for c, n in zip(classes, counts)},
    }
    (out / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def add_common_raw_options(parser):
    parser.add_argument("--input", required=True)
    parser.add_argument("--class-column", required=True)
    parser.add_argument("--numeric-columns", default="")
    parser.add_argument(
        "--numeric-method",
        choices=["lucs-kdd-style", "mdlp", "equal-width", "equal-frequency"],
        default="lucs-kdd-style",
    )
    parser.add_argument("--default-bins", type=int, default=5)
    parser.add_argument("--bins-by-column", default="")
    parser.add_argument("--mdlp-max-bins", type=int, default=MDLP_CAPACITY_DEFAULT)


def main():
    p = argparse.ArgumentParser(description="Prepare user datasets for Auto-DMC-AC")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("encode", help="Encode one raw CSV as a transactional file")
    add_common_raw_options(a)
    a.add_argument("--output", required=True)
    a.set_defaults(func=encode)

    b = sub.add_parser("outer-cv", help="Raw CSV -> fold-specific preprocessed outer CV partitions")
    add_common_raw_options(b)
    b.add_argument("--output-dir", required=True)
    b.add_argument("--outer-folds", type=int, default=10)
    b.add_argument("--seed", type=int, default=20260715)
    b.set_defaults(func=outer_cv)

    c = sub.add_parser("transactional-cv", help="Already-discrete transactional file -> reproducible partitions")
    c.add_argument("--input", required=True)
    c.add_argument("--output-dir", required=True)
    c.add_argument("--outer-folds", type=int, default=10)
    c.add_argument("--seed", type=int, default=20260715)
    c.set_defaults(func=transactional_cv)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
