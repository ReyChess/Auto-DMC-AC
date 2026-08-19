# Auto-DMC-AC v5.3

## Main change

v5.3 jointly selects the structural variant and the prior weight through the existing inner cross-validation:

- `global_p`
- `global_pn`
- `cns_p`
- `cns_pn`
- `lambda_prior` from a configurable grid

The predictive core remains unchanged:

- `K_MAX = 12`
- `BETA = 2`
- exact-first positive coverage
- one-missing-item positive fallback
- conservative exact negative veto in the near-tie region
- the same PN and CNS computational guards

## Computational design

Each structural variant is mined once per inner split. The resulting rules are reused for all lambda values. Therefore, increasing the number of lambda candidates does not multiply the number of mining runs. It only adds inexpensive classification rescoring.

## Default lambda grid

`0.05,0.10,0.15,0.20,0.30`

Override it with:

```bash
--lambda-grid 0,0.10,0.15,0.20,0.30
```

## Selection criterion

Candidates are ordered by:

1. mean inner macro-F1;
2. fewer rules;
3. lower total time;
4. lambda closest to the former fixed value 0.15;
5. positive-only before PN when tied;
6. global before CNS when tied.

## New output fields

- `lambda_prior` in `instance_log.csv` and `fold_variant_summary.csv`;
- `lambda_prior` in `selection_log.csv`;
- `selected_lambda_prior` in `dataset_summary.csv`;
- `lambda_grid` in `run_configuration.txt` and `auto_dmc_report.txt`.
