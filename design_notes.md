# Design Notes: Open Issues in the Super Learner UQ Pipeline

## 1. Absence of Legitimate Uncertainty Quantification Metrics

The current evaluation in `bii_super_learner.py` measures predictive *performance* (AUC, Brier, NLL, ECE, F1, MCC) but does not evaluate whether the uncertainty scores themselves are well-calibrated or informative. Specifically:

- **Spearman correlation between error and uncertainty** is computed (`Spearman_Err_Unc`), but this only tests rank correlation on a single split — it is not a proper UQ metric.
- **ECE** measures probability calibration, not uncertainty calibration. A model can have low ECE while assigning identical uncertainty scores to every sample.
- There is no **coverage–width curve** (CWC) or **Winkler score** to evaluate the quality of prediction intervals.
- There is no **selective prediction** evaluation with a proper abstention metric (e.g., AURC — Area Under the Risk-Coverage Curve).
- The `Avg_Entropy` column is a raw average of predictive entropy, which conflates aleatoric and epistemic components without testing whether either correlates with actual error regions.
- For the evidential model, the decomposed epistemic/aleatoric scores are output but never evaluated against held-out OOD indicators or error rates in a principled way (e.g., no AUROC of epistemic uncertainty vs. OOD label).

**What is needed:** At minimum, proper UQ evaluation should include:
- AURC or Risk-Coverage AUC: rank samples by uncertainty, measure accuracy on the retained fraction.
- Expected Calibration Error on the *uncertainty* axis, not just the probability axis.
- For OOD detection: AUROC of epistemic uncertainty scores using the val/test split as in-distribution / OOD labels respectively.

---

## 2. Per-Sample Adaptive Weights Are Not Compatible with Canonical Conformal Prediction

The `uncertainty_inverse_isotonic_reject_filtered` strategy assigns weights `w[k, n]` that vary **per sample** `n`. This violates the foundational assumption of standard (split) conformal prediction in two ways:

### 2a. Conformal Validity Requires Fixed Nonconformity Scores

In canonical split-CP, the calibration nonconformity scores `{s(x_i, y_i)}` and the test nonconformity score `s(x_test, y_test)` must be **exchangeable**. The p-value:

```
p(x_test) = |{i : s_i >= s_test}| / (n_cal + 1)
```

is valid only when the test score is computed by the same fixed scoring function as the calibration scores. When the ensemble weights `w[k, n]` depend on the test sample `n` itself (through the isotonic inverse-uncertainty scores and the per-method acceptance masks), the scoring function effectively changes for every test point. This breaks exchangeability.

### 2b. The Per-Method Acceptance Masks Introduce a Data-Dependent Scoring Rule

`_downweight_rejected_method_weights` zeroes out method `k` for sample `n` based on `base_accept_mask[k, n]`, which is itself derived from a conformal p-value evaluated at `(x_n, p_k(x_n))`. The final ensemble probability:

```
p_ens(x_n) = sum_k w_rf[k, n] * p_k(x_n)
```

is therefore a function of `x_n` that was not fixed before seeing `x_n`. Using this as input to a downstream conformal step (the CSA or standard credal set assignment) does not carry valid coverage guarantees — the test sample has already influenced its own scoring function.

### 2c. Lack of Coordination (Per the Design Discussion)

Beyond the CP compliance issue, the per-method masks are computed from K independent marginal score distributions with no cross-model coordination. A model is rejected based solely on its own historical score distribution, not on whether it is an outlier relative to the ensemble consensus on that sample. This means:

- Heterogeneous acceptance thresholds across models at the same nominal `α`.
- No distinction between a lone-dissenting model (where zeroing is appropriate) and unanimous ensemble disagreement (where zeroing and falling back to uniform hides genuine uncertainty).

---

## 3. Summary of Open Issues

| Issue | Severity | Affected component |
|---|---|---|
| No proper UQ evaluation metrics (AURC, OOD-AUROC) | High | All model evaluation |
| ECE/Spearman are not UQ metrics | Medium | `_evaluate_like_main_active` |
| Per-sample weights break CP exchangeability | High | `uncertainty_inverse_isotonic_reject_filtered` |
| Per-method CP masks not coordinated across models | Medium | `_downweight_rejected_method_weights` |
| Weighted CP weights (cp_w_val/test) are all-ones by default | Medium | Standard CP path |
| CSA receives no covariate-shift correction weights | High | `_csa_fit` / `_csa_predict` |

---

## 4. Proposed Directions

- **For UQ metrics**: add AURC, compute AUROC of epistemic uncertainty using val-as-ID / test-as-OOD, and report coverage at multiple uncertainty quantile cutoffs.
- **For CP compliance**: if adaptive ensemble weights are desired, they must be fixed (i.e., depend only on the calibration data, not on the test point). One option is to replace per-sample weights with a fixed global weight vector learned entirely on the val set.
- **For coordination**: if per-sample masking is retained for exploratory purposes, replace K independent marginal CP checks with a consensus gate (zero model k only if a majority of other models accept the sample) or CSA-based attribution (identify which model drives the joint T-score violation).
