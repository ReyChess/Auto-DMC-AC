#!/usr/bin/env python3
"""Validate Auto-DMC-AC's exported prediction explanations."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


TOL = 1e-9
BETA = 2.0


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("explanation_dir", type=Path)
    args = parser.parse_args()
    directory = args.explanation_dir

    required = {
        "global_rule_model.csv",
        "prediction_summary.csv",
        "prediction_class_scores.csv",
        "prediction_rule_trace.csv",
        "prediction_used_cars.csv",
        "prediction_explanations.txt",
        "trace_verification.csv",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    assert not missing, f"Missing explanation outputs: {missing}"

    summary = rows(directory / "prediction_summary.csv")
    verification = rows(directory / "trace_verification.csv")
    class_scores = rows(directory / "prediction_class_scores.csv")
    rule_rows = rows(directory / "prediction_rule_trace.csv")
    used_rows = rows(directory / "prediction_used_cars.csv")
    assert summary and len(summary) == len(verification)
    assert all(row["instance_items"] != "" for row in summary)
    assert all(row["trace_verified"] == "1" for row in summary)
    assert all(row["verified"] == "1" for row in verification)

    positive = defaultdict(list)
    negative = defaultdict(list)
    for row in rule_rows:
        key = (row["outer_fold"], row["instance_id"], row["class_label"])
        (positive if row["evidence_type"] == "positive" else negative)[key].append(row)

    for evidence in positive.values():
        retained = [row for row in evidence if row["retained_for_decision"] == "1"]
        assert len(retained) <= 12
        assert [int(row["evidence_rank"]) for row in retained] == list(range(1, len(retained) + 1))
    for evidence in negative.values():
        assert sum(row["retained_for_decision"] == "1" for row in evidence) <= 2

    retained_keys = {
        (row["outer_fold"], row["instance_id"], row["evidence_type"], row["class_label"], row["rule_id"])
        for row in rule_rows if row["retained_for_decision"] == "1"
    }
    used_keys = {
        (row["outer_fold"], row["instance_id"],
         "positive" if row["rule_role"] == "positive_decision" else "negative",
         row["consequent_class"], row["rule_id"])
        for row in used_rows
    }
    assert used_keys == retained_keys
    assert len(used_rows) == len(retained_keys)
    required_used_values = {
        "antecedent", "coverage_mode", "retained_rank", "netconf",
        "final_class_score", "negative_class_score", "negative_evaluated",
        "veto_condition_met", "veto_changed_prediction",
    }
    assert all(all(row[name] != "" for name in required_used_values) for row in used_rows)

    checked_scores = 0
    for row in class_scores:
        if row["positive_candidate"] != "1":
            continue
        key = (row["outer_fold"], row["instance_id"], row["class_label"])
        retained = [r for r in positive[key] if r["retained_for_decision"] == "1"]
        assert retained
        numerator = sum(float(r["weighted_contribution"]) for r in retained)
        denominator = sum(float(r["rank_weight"]) for r in retained)
        raw = numerator / denominator
        assert close(raw, float(row["raw_positive_score"]))

        k = len(retained)
        mu0 = float(row["mu0_positive"])
        shrunken = (raw * k + BETA * mu0) / (k + BETA)
        assert close(shrunken, float(row["shrunken_positive_score"]))

        prior = float(row["class_prior"])
        lambda_prior = float(row["lambda_prior"])
        prior_contribution = lambda_prior * math.log(prior)
        assert close(prior_contribution, float(row["prior_contribution"]))
        assert close(shrunken + prior_contribution, float(row["final_score"]))
        checked_scores += 1

    assert checked_scores > 0
    print(
        f"PASS: {len(summary)} predictions reconstructed; "
        f"{checked_scores} candidate-class score calculations verified."
    )


if __name__ == "__main__":
    main()
