# Auto-DMC-AC v5.4.1

Boundary-correctness patch for positive Netconf admission.

- Positive CARs now require `Netconf(A=>c) > min_netconf_pos`. With the
  experimental default `min_netconf_pos=0`, statistically independent
  antecedent/class pairs (`Netconf=0`) are no longer stored as positive
  evidence.
- The same strict comparison is applied in both the global-support and
  class-normalized-support miners, including the auxiliary positive-candidate
  path.
- Negative CAR admission is unchanged and remains strictly `Netconf(A=>c)<0`.
- No support policy, automatic-selection grid, predictor, shrinkage, prior,
  top-K rule, PN veto, or computational-guard setting was changed.

Audit motivating and validating this patch: across the 150 frozen final models
of the 15-dataset benchmark, only 3 of 1,116,632 final rules had exactly zero
Netconf. They covered 114 test instances but removing them changed 0 of 53,084
out-of-fold predictions (44,152 correct predictions before and after).

The complete nested experiment was then rerun with strict `Netconf>0` admission
for all 15 datasets. Across all 150 outer folds, the selected structural
variant, sigma and lambda were identical to the original run, and every outer
accuracy and macro-F1 value was preserved. The strict run therefore retains
81.862664796% unweighted mean accuracy, 0.714773679860 unweighted mean macro-F1,
and 44,152/53,084 pooled correct predictions. See `ETA0_STRICT_AUDIT.md`.
