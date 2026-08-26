from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from prepare_user_dataset import (
    entropy, mdlp_recursive, equal_width_edges, equal_frequency_edges,
    NamespacePlan, FoldEncoder, parse_bins_spec
)


def test_entropy():
    assert round(entropy([0, 0, 1, 1]), 6) == 1.0


def test_mdlp_finds_clear_split():
    x = np.array([1, 2, 3, 4, 10, 11, 12, 13], dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    cuts = mdlp_recursive(x, y)
    assert len(cuts) == 1
    assert 4 < cuts[0] < 10


def test_equal_width_count():
    assert len(equal_width_edges([0, 10], 5)) - 1 == 5


def test_equal_frequency_count_with_unique_data():
    assert len(equal_frequency_edges(np.arange(100), 5)) - 1 == 5


def test_class_is_last_and_ids_are_compact():
    df = pd.DataFrame({"x": [1, 2, 8, 9], "cat": ["a", "b", "a", "b"], "class": ["N", "N", "Y", "Y"]})
    nums = ["x"]
    bins = parse_bins_spec("", nums, 5)
    ns = NamespacePlan(df, "class", nums, "mdlp", bins, 32)
    enc = FoldEncoder("class", nums, "mdlp", bins, ns).fit(df)
    rows = enc.transform(df)
    class_ids = set(ns.class_items.values())
    assert all(r[-1] in class_ids for r in rows)
    assert ns.max_item_id < 1000


def test_mdlp_metadata_has_numeric_edges():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 10, 11, 12, 13], "class": ["N"] * 4 + ["Y"] * 4})
    nums = ["x"]
    bins = parse_bins_spec("", nums, 5)
    ns = NamespacePlan(df, "class", nums, "mdlp", bins, 32)
    enc = FoldEncoder("class", nums, "mdlp", bins, ns).fit(df)
    assert len(enc.metadata()["feature_models"]["x"]["edges"]) >= 3


def test_class_ids_stable_across_fold_encoders():
    df = pd.DataFrame({"x": [1, 2, 8, 9], "class": ["N", "N", "Y", "Y"]})
    nums = ["x"]
    bins = parse_bins_spec("", nums, 5)
    ns = NamespacePlan(df, "class", nums, "mdlp", bins, 32)
    e1 = FoldEncoder("class", nums, "mdlp", bins, ns).fit(df.iloc[[0, 1, 2]])
    e2 = FoldEncoder("class", nums, "mdlp", bins, ns).fit(df.iloc[[1, 2, 3]])
    assert e1.namespace.class_items == e2.namespace.class_items
