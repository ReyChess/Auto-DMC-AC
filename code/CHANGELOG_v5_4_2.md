# Auto-DMC-AC v5.4.2

## Prior-adjusted base-decision consistency

The prior-adjusted positive base class is now selected over **all classes with
positive evidence**:

`argmax_c [ S_c^+(t) + lambda_prior * log(pi(c)) ]`, for `k_c(t) > 0`.

In v5.4.1, the ordinary and cached lambda-grid classifiers used this global
decision when the unadjusted positive winner was unique, but recomputed the
prior-adjusted winner only inside the unadjusted near-tie set when that set had
more than one class.  That hybrid behavior was inconsistent with the formal
classifier definition.

The v5.4.2 correction is applied in both `classify` and
`classify_lambda_grid`, so inner model selection and outer evaluation use the
same decision rule.  The unadjusted near-tie set remains lambda-independent and
continues to delimit/diagnose the optional negative-evidence stage; it no
longer restricts the positive base argmax.

The strict positive Netconf admission introduced in v5.4.1 is unchanged.
