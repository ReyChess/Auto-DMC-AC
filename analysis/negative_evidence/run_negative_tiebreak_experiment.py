#!/usr/bin/env python3
"""Frozen positive model + parameter-free negative tie-break experiment.

For each outer fold, reuse the already-selected P-only support mode, sigma,
lambda and positive rule base.  If the original positive classifier produces
a near tie (epsilon=.005), exact negative rules are mined at the same
support/mode and the candidate class with the smallest existing negative score
is selected.  Equal negative evidence keeps the positive winner.  No outer
test label participates in the decision.
"""
import csv
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_predictor_ablation"
MINER = ROOT / "experiment_bin" / "dmc_miner_unified"
OUT = ROOT / "negative_tiebreak_experiment"
OUT.mkdir(exist_ok=True)

EPS = 0.005
KMAX = 12
BETA = 2.0
KNEG = 2
BETA_NEG = 15.0
TIMEOUT = 120


def read_dat(path):
    rows = []
    with open(path) as f:
        for line in f:
            vals = [int(x) for x in line.split()]
            if vals:
                rows.append(vals)
    return rows


def read_rules(path):
    by_class = defaultdict(list)
    with open(path) as f:
        for line in f:
            z = line.split()
            if not z:
                continue
            n = int(z[0])
            items = [int(x) for x in z[1:1+n]]
            raw_class = items[-1]
            antecedent = frozenset(items[:-1])
            netconf = float(z[1+n+3])
            by_class[abs(raw_class)].append((antecedent, netconf, raw_class < 0))
    for c in by_class:
        by_class[c].sort(key=lambda r: (-len(r[0]), -r[1]))
    return by_class


def pos_score(vals, mu0):
    if not vals:
        return -1e300
    vals = vals[:KMAX]
    weights = [1.0 / (j + 1) for j in range(len(vals))]
    avg = sum(w*v for w, v in zip(weights, vals)) / sum(weights)
    k = len(vals)
    return (avg*k + BETA*mu0) / (k+BETA)


def neg_score(vals):
    strengths = sorted((-x for x in vals if -x > 0), reverse=True)
    if not strengths:
        return 0.0
    used = min(len(strengths), KNEG)
    return strengths[0] * used / (used+BETA_NEG)


def macro_f1(y, pred, labels):
    s = 0.0
    for c in labels:
        tp = sum(a == c and b == c for a, b in zip(y, pred))
        fp = sum(a != c and b == c for a, b in zip(y, pred))
        fn = sum(a == c and b != c for a, b in zip(y, pred))
        den = 2*tp + fp + fn
        s += 2*tp/den if den else 0.0
    return s/len(labels)


def same(a, b):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


fold_rows = []
event_rows = []

for dsdir in sorted(p for p in RESULTS.iterdir() if p.is_dir() and (p/"selection_log.csv").exists()):
    ds = dsdir.name
    sel = pd.read_csv(dsdir/"selection_log.csv")
    guards = pd.read_csv(dsdir/"pn_guard.csv")
    ilog = pd.read_csv(dsdir/"instance_log.csv")
    ilog = ilog[ilog.phase == "outer"]

    for outer in range(1, 11):
        chosen = sel[(sel.outer_fold == outer) & (sel.selected == 1)]
        if len(chosen) != 1:
            raise RuntimeError(f"{ds} fold {outer}: expected one selected model, got {len(chosen)}")
        chosen = chosen.iloc[0]
        if chosen.variant not in ("global_p", "cns_p"):
            raise RuntimeError(f"{ds} fold {outer}: selected model is not P-only: {chosen.variant}")
        support = "cns" if chosen.variant == "cns_p" else "global"
        sigma = float(chosen.sigma)
        lam = float(chosen.lambda_prior)

        gm = guards[(guards.outer_fold == outer) & (guards.support_mode == support)]
        gm = gm[np.isclose(gm.sigma.astype(float), sigma, atol=1e-12, rtol=0)]
        eligible = len(gm) > 0 and bool((gm.class_guard_pass.astype(int) & gm.budget_guard_pass.astype(int)).max())
        if not eligible:
            fold_rows.append(dict(dataset=ds, outer_fold=outer, status="INELIGIBLE",
                                  support=support, sigma=sigma, lambda_prior=lam))
            continue

        final_dir = dsdir/f"outer_{outer}"/"final"
        train_path, test_path = final_dir/"train.dat", final_dir/"test.dat"
        p_rules_path = final_dir/"rules.dat"
        outdir = OUT/ds/f"outer_{outer}"
        outdir.mkdir(parents=True, exist_ok=True)
        pn_rules_path = outdir/"rules_pn.dat"
        miner_log = outdir/"miner.log"
        if not pn_rules_path.exists():
            cmd = [str(MINER), str(train_path), f"{sigma:.12g}", str(pn_rules_path), support, "pn", "0.0"]
            try:
                with open(miner_log, "w") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT, check=True)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                pn_rules_path.unlink(missing_ok=True)
                fold_rows.append(dict(dataset=ds, outer_fold=outer, status=type(e).__name__.upper(),
                                      support=support, sigma=sigma, lambda_prior=lam))
                continue

        train, test = read_dat(train_path), read_dat(test_path)
        p_rules, pn_rules = read_rules(p_rules_path), read_rules(pn_rules_path)
        counts = Counter(r[-1] for r in train)
        labels = sorted(counts)
        prior = {c: counts[c]/len(train) for c in labels}
        logfold = ilog[ilog.outer_fold == outer].sort_values("instance_id")
        if len(logfold) != len(test):
            raise RuntimeError(f"{ds} fold {outer}: instance-log/test size mismatch")
        base_preds = logfold.predicted_class.astype(int).tolist()
        final_preds = list(base_preds)
        truths = [r[-1] for r in test]
        changes = corrections = harms = neg_events = 0

        for idx, (row, lr) in enumerate(zip(test, logfold.itertuples(index=False))):
            if int(lr.near_tie) != 1:
                continue
            inst = set(row[:-1])
            pos = {c: [] for c in labels}
            exact_total = 0
            for c in labels:
                for ant, nc, neg in p_rules.get(c, []):
                    if not neg and ant.issubset(inst):
                        pos[c].append(nc); exact_total += 1
            if exact_total == 0:
                for c in labels:
                    for ant, nc, neg in p_rules.get(c, []):
                        if not neg and len(ant-inst) == 1:
                            pos[c].append(nc)
            all_pos = [x for c in labels for x in pos[c]]
            if not all_pos:
                continue
            mu0 = sum(all_pos)/len(all_pos)
            scores = {c: pos_score(pos[c], mu0) for c in labels}
            best_pos = max(scores.values())
            tied = [c for c in labels if scores[c] > -1e200 and best_pos-scores[c] <= EPS]
            if len(tied) <= 1:
                raise RuntimeError(f"{ds} fold {outer} instance {idx+1}: logged near-tie not reconstructed")
            base = max(tied, key=lambda c: (scores[c] + lam*math.log(prior[c]), -c))
            if base != int(lr.predicted_class):
                raise RuntimeError(f"{ds} fold {outer} instance {idx+1}: base mismatch {base}!={lr.predicted_class}")

            negvals = {}
            activated = 0
            for c in tied:
                vals = [nc for ant, nc, neg in pn_rules.get(c, []) if neg and ant.issubset(inst)]
                activated += len(vals)
                negvals[c] = neg_score(vals)
            if activated == 0:
                continue
            neg_events += 1
            min_neg = min(negvals.values())
            least = [c for c in tied if abs(negvals[c]-min_neg) <= 1e-12]
            alt = base if base in least else max(least, key=lambda c: (scores[c] + lam*math.log(prior[c]), -c))
            changed = alt != base
            if changed:
                changes += 1
                final_preds[idx] = alt
                corrections += int(base != truths[idx] and alt == truths[idx])
                harms += int(base == truths[idx] and alt != truths[idx])
            event_rows.append(dict(dataset=ds, outer_fold=outer, instance=idx+1, true_class=truths[idx],
                                   base_class=base, tiebreak_class=alt, changed=int(changed),
                                   base_correct=int(base==truths[idx]), tiebreak_correct=int(alt==truths[idx]),
                                   tie_size=len(tied), negative_rules_activated=activated,
                                   base_negative_score=negvals[base], selected_negative_score=negvals[alt]))

        base_correct = sum(a == b for a,b in zip(truths,base_preds))
        final_correct = sum(a == b for a,b in zip(truths,final_preds))
        fold_rows.append(dict(dataset=ds, outer_fold=outer, status="OK", support=support, sigma=sigma,
                              lambda_prior=lam, n=len(test), neg_tie_events=neg_events, changes=changes,
                              corrections=corrections, harms=harms, base_accuracy=base_correct/len(test),
                              tiebreak_accuracy=final_correct/len(test),
                              base_macro_f1=macro_f1(truths,base_preds,labels),
                              tiebreak_macro_f1=macro_f1(truths,final_preds,labels)))
        pn_rules_path.unlink(missing_ok=True)
        print(ds, outer, "events", neg_events, "changes", changes, "fix", corrections, "harm", harms, flush=True)

fold = pd.DataFrame(fold_rows)
events = pd.DataFrame(event_rows)
fold.to_csv(OUT/"fold_results.csv", index=False)
events.to_csv(OUT/"tiebreak_events.csv", index=False)

ok = fold[fold.status == "OK"].copy()
complete = [d for d,g in ok.groupby("dataset") if len(g) == 10]
ds = ok[ok.dataset.isin(complete)].groupby("dataset", as_index=False).agg(
    base_accuracy=("base_accuracy","mean"), tiebreak_accuracy=("tiebreak_accuracy","mean"),
    base_macro_f1=("base_macro_f1","mean"), tiebreak_macro_f1=("tiebreak_macro_f1","mean"),
    events=("neg_tie_events","sum"), changes=("changes","sum"), corrections=("corrections","sum"), harms=("harms","sum"))
ds["delta_accuracy"] = ds.tiebreak_accuracy-ds.base_accuracy
ds["delta_macro_f1"] = ds.tiebreak_macro_f1-ds.base_macro_f1
ds.to_csv(OUT/"complete_dataset_results.csv", index=False)

summary = []
for metric in ("accuracy","macro_f1"):
    diff = ds[f"tiebreak_{metric}"]-ds[f"base_{metric}"]
    nz = diff[np.abs(diff)>1e-15]
    p = wilcoxon(diff, alternative="two-sided").pvalue if len(nz) else 1.0
    summary.append(dict(metric=metric, datasets=len(ds), base_mean=ds[f"base_{metric}"].mean(),
                        tiebreak_mean=ds[f"tiebreak_{metric}"].mean(), delta_mean=diff.mean(),
                        wins=int((diff>1e-15).sum()), ties=int((abs(diff)<=1e-15).sum()),
                        losses=int((diff<-1e-15).sum()), wilcoxon_p=p))
pd.DataFrame(summary).to_csv(OUT/"statistical_summary.csv", index=False)

print("\nCOMPLETE DATASETS", complete)
print(ds.to_string(index=False))
print("\nSUMMARY")
print(pd.DataFrame(summary).to_string(index=False))
print("\nALL ELIGIBLE FOLDS", len(ok), "events", int(ok.neg_tie_events.sum()), "changes", int(ok.changes.sum()),
      "corrections", int(ok.corrections.sum()), "harms", int(ok.harms.sum()))
