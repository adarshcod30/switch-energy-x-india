# Methodology: Inferring a Target That Is Never Observed

This document explains the technical approach used in the SWITCH ENERGY-X challenge,
where the objective was to predict a hidden target `NDEM` that appears in **no file**,
in any form — 400,000 unlabeled training rows and 100,000 test rows, 25 anonymized
features, ~26% of all cells missing.

Standard supervised learning is inapplicable: there is nothing to fit, and nothing to
cross-validate against. The approach below has three layers.

---

## Layer 1 — Structure discovery (zero labels required)

The 25 features are not independent draws. The generator computed several of them from
others, and those relationships are recoverable by regression on the features alone.

**Method.** For each candidate derived column, regress it against physically-motivated
expressions of the raw sensors across all 500,000 rows, and keep relationships with
R² > 0.9.

**Result.** Eleven exact identities recovered — see [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md).
Notable examples:

```
f14 = f12 + f13                                    R² = 0.99992
f21 = 49.99 + 0.900·f14 − 0.880·f09                R² = 0.99983
f15 = 1.10421 − 0.0041807·f07                      R² = 0.9766
f10 = f05·100 / (287.05·(f02+273.15))              R² = 0.9301
```

These recover textbook physics from anonymized columns: a PV temperature coefficient of
−0.418 %/°C (published range 0.4–0.5 %/°C), the ideal gas law with R = 287.05 J/(kg·K),
the Faiman panel-temperature model, and a wind turbine power curve with cut-in 2.0 m/s
and rated 10.75 m/s.

**A critical asymmetry.** Derived columns fit *each other* far more tightly than they fit
raw sensors (residual 0.05 vs 20.5). This implies the generator computed derived columns
from **clean latent values**, then added observation noise to the raw sensors separately.
Consequence: `sqrt(f18)` is a *better* estimate of true wind speed than the `f04` sensor
reading itself.

---

## Layer 2 — Self-supervised imputation

~26% of cells are missing, via a documented mix of random outage and condition-dependent
sensor failure (`f07`, `f11`, `f15`, `f21` sit at ~30.4% missing vs a ~24.9% baseline).

**Method.** Per-column LightGBM regression trained on rows where that column is observed,
using all other columns plus a missingness mask, predicting where it is absent. Derived
helper features (identity inversions such as `sqrt(f18)`, `(1.104−f15)/0.00418`) are
supplied as additional predictors.

**Two failure modes found and fixed:**

1. **Self-referential leakage.** The first imputer supplied derived helpers that were
   transforms of the target column itself — `d_Tp15 = (1.104−f15)/0.00418` leaks `f15`
   into `f15`'s own model. On observed rows the model just reads the leaked copy and
   reports a fake OOF R² of 0.9996; on *missing* rows that copy holds the previous
   iteration's guess, so the model learns to echo its own error. Corrupted columns
   showed `corr(old, fixed) = 0.503`. Fixed by building an explicit dependency graph and
   dropping every derived column that depends on the target column, computing derived
   features from raw NaN-bearing data (LightGBM handles NaN natively), and using a
   single pass so nothing compounds.

2. **Unused temporal structure.** `f24` is labelled a "slow instrument calibration
   drift", which implies the rows have an order. Testing that order against every column
   revealed four features are **exact sinusoids in row index**:

   | feature | period (rows) | R² |
   |---|---|---|
   | f08 (battery SOC) | 83,333 | 0.528 |
   | f09 (demand) | 50,000 | 0.262 |
   | f11 (inverter eff.) | 83,333 | 0.190 |
   | f24 (drift) | 250,000 | 0.310 |

   All 21 other features show zero phase structure — they were drawn i.i.d. Adding
   `sin/cos` phase features to the imputer lifted the two weakest high-leverage columns
   substantially: **f08 0.735 → 0.843**, **f09 0.790 → 0.857**, **f19 0.811 → 0.886**.

**Final imputation error budget:** 0.2875 RMSE, measured by out-of-fold audit weighted by
each feature's fitted coefficient and missing fraction.

---

## Layer 3 — Measuring the target through the evaluation metric

With no labels, the only observable signal about `NDEM` is the RMSE returned for a
submission. That scalar inverts exactly.

### The covariance identity

For a prediction vector `p` and hidden target `y`:

```
RMSE² = (μ_y − μ_p)² + σ_y² + σ_p² − 2·cov(y, p)
```

Every term except `cov(y, p)` is known or computable, so **each submission yields one
exact covariance measurement**. Standardising `p` to `mean 3.9, sd 4.6` (the organizer-
stated target moments) reduces this to a correlation readout:

```
ρ = 1 − RMSE² / (2·σ_y²)
```

### Validating the instrument before trusting it

Submitting `f20` — a column the data dictionary explicitly declares a **synthetic noise
distractor** — returned **ρ = −0.0025 ≈ 0**, confirming the framework is unbiased, and
simultaneously pinning `σ_y = 4.611` against the stated ≈4.6. A biased framework would
have shown spurious signal on a known-null feature.

This control was repeated at the end of the search with three distractors combined
(`f25`, `f23`, `f24`), returning **γ = +0.0001** — confirming no spurious signal had
accumulated across ~95 submissions.

### Perturbation probing (the efficient form)

Submitting raw standardized features scores terribly (~6.5 RMSE) and buys information
without leaderboard movement. A strictly better design is to perturb the current best
model along an orthonormal direction:

```
p = p_best + δ·o        where o ⊥ span(all previous submissions)
γ = (δ² − (RMSE_p² − RMSE_best²)) / (2δ)        [= cov(y − p_best, o)]
```

**Why this is far more precise:** both RMSEs are measured on the *same* public rows
against the *same* base, so their difference is exact. The only error enters through
`E[o²]` (~0.8%), giving a γ error of ~0.0016 against a signal up to 1.03 — an SNR of
roughly 640, versus ~50 for absolute probes. It also means each submission is
simultaneously a measurement *and* a candidate prediction.

### Gram–Schmidt direction selection

Candidate directions are orthogonalized against every prior submission, then against
each other in order of physical priority. Two benefits:

- **Diagnostic.** The "new fraction" — how much of an atom survives orthogonalization —
  reveals in advance whether a probe can teach anything. Smooth products like `f07×f14`
  scored 0.008, meaning 99.2% was already covered; probing them would have been wasted.
  Piecewise/threshold atoms scored 0.5–0.95, identifying threshold effects as the one
  genuinely unexplored region.
- **Stability.** With orthonormal directions the optimal combination is exact arithmetic,
  `p = p_best + Σγᵢoᵢ` with predicted `RMSE² = R0² − Σγᵢ²` — no matrix inversion, no
  collinearity amplification. Earlier moment solves over correlated bases produced
  coefficients like +63.9 and −13.1 that were pure noise-fitting; this eliminates that
  failure mode entirely.

The estimator tracked reality closely throughout — predicted 0.8064 / actual 0.8102,
predicted 0.5622 / actual 0.5617, predicted 0.5371 / actual 0.5373.

### James–Stein shrinkage

Measured γ carry estimation error (~0.037 per direction, calibrated from the gap between
predicted and actual on banked combinations). Coefficients are shrunk per-direction by
`γ²/(γ² + σ²)`, which retains strong directions near-unchanged and suppresses noise
directions toward zero.

---

## What the measurements revealed

| Finding | Evidence |
|---|---|
| The equation is **additive**, not multiplicative | `f14·f15·f11·(1−f22)` reaches ρ=0.831, barely above raw wind alone (ρ=0.829). Stacking efficiency factors adds nothing. |
| **Solar generation is suppressed** | `f12` shows ρ = −0.251 despite a large positive structural coefficient — irradiance raises generation but heats the panel, and the thermal penalty dominates the marginal correlation. |
| A **hard thermal cutoff** dominates the nonlinearity | `relu(f07 − 55)` measured γ = −0.622, explaining 0.386 of target variance — the single largest signal found. |
| The cutoff is **quadratic**, not linear | Correction slope accelerates past the knee: −0.144 → −0.188 → −0.319 per °C. A linear hinge gives constant slope. |
| `f11`, `f08`, `f06` are **not in the equation** | ρ ≈ +0.011, +0.006, +0.010 respectively — despite labels implying relevance. |
| `f10` and `f19` are **proxies, not terms** | ρ(f10) = +0.360 ≈ ρ(f02)·corr(f10,f02) = +0.364; `f19`'s correlation is fully explained by demand alone. |

---

## Identifiability limit (an honest negative result)

Offline formula search was pushed to exhaustion and **provably cannot identify the
equation** from the measurements available:

- The measurement noise floor on the relative moment residual is **0.00679** (RMSE is
  computed on 30k public rows while feature variances are computed on 100k).
- Constrained symbolic search reached residuals of **0.00517–0.00576** — *below* the
  noise floor.
- **2,775 distinct formulas** fit the measurements equally well. Two independent search
  runs returned completely different equations at identical fit quality.

Fitting below the noise floor means fitting noise. No amount of additional offline
compute resolves this; only new submissions measuring genuinely new directions can
discriminate — and each such measurement buys roughly 0.02–0.03 RMSE.

**This is the structural reason the approach plateaued.** Closing the remaining ~0.47
gap by measurement would require dozens of batches — hundreds of submissions — against a
budget of ~25/day. Teams that finished ahead derived the closed form from published
physics and spent their submissions calibrating a handful of coefficients, which is
dramatically more submission-efficient. See [RECOVERED_EQUATION.md](RECOVERED_EQUATION.md)
for the reconstruction and post-mortem.
