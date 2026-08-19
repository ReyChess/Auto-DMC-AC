#!/usr/bin/env python3
import csv
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_policy_selector"
OUT = ROOT / "neg_conflict"
MINER = ROOT / "experiment_bin" / "dmc_miner_unified"
EVALUATOR = ROOT / "experiment_bin" / "auto_dmc_ac_conflict"
TIMEOUT_SEC = 90


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def same_float(a, b):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)


OUT.mkdir(parents=True, exist_ok=True)
all_instances = []
fold_rows = []

for ds_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir() and (p / "selection_log.csv").exists()):
    dataset = ds_dir.name
    selections = read_csv(ds_dir / "selection_log.csv")
    guards = read_csv(ds_dir / "pn_guard.csv")
    for outer in range(1, 11):
        candidates = [r for r in selections
                      if int(r["outer_fold"]) == outer and r["variant"] in ("global_p", "cns_p")]
        if not candidates:
            fold_rows.append({"dataset": dataset, "outer_fold": outer, "status": "NO_P_CANDIDATE",
                              "variant": "", "sigma": "", "lambda_prior": "", "instances": 0,
                              "near_ties": 0, "conflicts": 0, "errors": 0, "conflict_errors": 0})
            continue
        best = min(candidates, key=lambda r: int(r["rank"]))
        support = "cns" if best["variant"] == "cns_p" else "global"
        sigma = best["sigma"]
        lam = best["lambda_prior"]

        matching_guards = [g for g in guards
                           if int(g["outer_fold"]) == outer and g["support_mode"] == support
                           and same_float(g["sigma"], sigma)]
        # A completed PN candidate at the same outer fold/support/sigma is
        # stronger evidence of guard eligibility than the guard CSV itself.
        # This also makes resumed runs robust when a guard log was truncated.
        pn_variant = support + "_pn_direct"
        completed_pn_candidate = any(int(r["outer_fold"]) == outer and r["variant"] == pn_variant
                                     and same_float(r["sigma"], sigma) for r in selections)
        if not (any(g["decision"] == "EVALUATE" for g in matching_guards) or completed_pn_candidate):
            reason = matching_guards[0]["reason"] if matching_guards else "pn_guard_not_evaluated"
            fold_rows.append({"dataset": dataset, "outer_fold": outer, "status": "SKIP:" + reason,
                              "variant": best["variant"], "sigma": sigma, "lambda_prior": lam, "instances": 0,
                              "near_ties": 0, "conflicts": 0, "errors": 0, "conflict_errors": 0})
            continue

        final_dir = ds_dir / f"outer_{outer}" / "final"
        train = final_dir / "train.dat"
        test = final_dir / "test.dat"
        fold_out = OUT / dataset / f"outer_{outer}"
        fold_out.mkdir(parents=True, exist_ok=True)
        rules = fold_out / "rules_pn.dat"
        miner_log = fold_out / "miner.log"
        instances_path = fold_out / "detector_instances.csv"

        try:
            if not instances_path.exists():
                cmd = [str(MINER), str(train), sigma, str(rules), support, "pn", "0.0"]
                with miner_log.open("w") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_SEC, check=True)
                ev = [str(EVALUATOR), "--detector-train", str(train), "--detector-test", str(test),
                      "--detector-rules", str(rules), "--detector-output", str(instances_path),
                      "--detector-lambda", lam, "--detector-tie-eps", "0.005"]
                subprocess.run(ev, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=TIMEOUT_SEC, check=True)
        except subprocess.TimeoutExpired:
            fold_rows.append({"dataset": dataset, "outer_fold": outer, "status": "SKIP:runtime_timeout",
                              "variant": best["variant"], "sigma": sigma, "lambda_prior": lam, "instances": 0,
                              "near_ties": 0, "conflicts": 0, "errors": 0, "conflict_errors": 0})
            if rules.exists(): rules.unlink()
            continue
        except subprocess.CalledProcessError:
            fold_rows.append({"dataset": dataset, "outer_fold": outer, "status": "SKIP:backend_error",
                              "variant": best["variant"], "sigma": sigma, "lambda_prior": lam, "instances": 0,
                              "near_ties": 0, "conflicts": 0, "errors": 0, "conflict_errors": 0})
            if rules.exists(): rules.unlink()
            continue

        rows = read_csv(instances_path)
        n_conflict = n_near = n_err = n_conflict_err = 0
        for r in rows:
            conflict = int(r["near_tie"]) == 1 and int(r["negative_covered"]) == 1
            error = int(r["correct"]) == 0
            n_near += int(r["near_tie"])
            n_conflict += int(conflict)
            n_err += int(error)
            n_conflict_err += int(conflict and error)
            all_instances.append({"dataset": dataset, "outer_fold": outer, "instance": r["instance"],
                                  "true_class": r["true_class"], "predicted_class": r["predicted_class"],
                                  "correct": r["correct"], "near_tie": r["near_tie"],
                                  "negative_covered": r["negative_covered"],
                                  "negative_rules_activated": r["negative_rules_activated"],
                                  "neg_conflict": int(conflict)})
        fold_rows.append({"dataset": dataset, "outer_fold": outer, "status": "OK",
                          "variant": best["variant"], "sigma": sigma, "lambda_prior": lam, "instances": len(rows),
                          "near_ties": n_near, "conflicts": n_conflict, "errors": n_err,
                          "conflict_errors": n_conflict_err})
        # PN rule bases are reproducible intermediates and can be very large.
        rules.unlink(missing_ok=True)
        print(dataset, outer, best["variant"], sigma, lam, "n=", len(rows), "conflicts=", n_conflict,
              "conflict_errors=", n_conflict_err, flush=True)

write_csv(OUT / "fold_summary.csv", fold_rows,
          ["dataset", "outer_fold", "status", "variant", "sigma", "lambda_prior", "instances",
           "near_ties", "conflicts", "errors", "conflict_errors"])
write_csv(OUT / "instances.csv", all_instances,
          ["dataset", "outer_fold", "instance", "true_class", "predicted_class", "correct", "near_tie",
           "negative_covered", "negative_rules_activated", "neg_conflict"])
