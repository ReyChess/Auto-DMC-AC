# Auto-DMC-AC v5.3.2 — deterministic selection tie-break

v5.3.2 preserves the mining, rule generation, evidence aggregation, shrinkage,
exact-first partial coverage, negative-veto logic, feasibility guards, and
lambda-grid cache of v5.3.1. The only algorithmic change is the final resolution
of exact inner-selection ties.

Selection order in v5.3.2:

1. Higher mean inner macro-F1.
2. Fewer mean stored rules.
3. Within the same structural candidate, lambda closest to the historical
   reference `lambda_prior=0.15`.
4. If two lambda values of the same structural candidate are equidistant from
   0.15, select the smaller lambda (less prior influence).
5. For remaining structural ties, prefer positive-only over PN, then global
   over CNS, then fixed lexical candidate order.

Execution time remains recorded in the logs as a computational measure but no
longer participates in model selection.

This change removes dependence on runtime noise in exact selection ties and
matches the deterministic selection procedure described in the final article.
