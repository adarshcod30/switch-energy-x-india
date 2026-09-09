# NDEM: Reconstructed Equation and Post-Mortem

**Final result: 12th of 63 teams, private RMSE 0.48008 (public 0.46441).**
**Winner: DP Team, private RMSE ≈0.292.**

This document lays out everything we actually measured about the hidden target during
the competition, the best-guess closed-form equation those measurements imply, and an
honest account of why we didn't get further.

---

## 1. What we know for certain (recovered with zero labels, from the data alone)

These are exact algebraic identities among the *features*, fit on all 500,000 rows.
None of this required any target information — it's pure structure discovery.

| Identity | Fit | Physical meaning |
|---|---|---|
| `f14 = f12 + f13` | R² = 0.99992 | Gross generation = solar + wind |
| `f21 = 49.99 + 0.900·f14 − 0.880·f09` | R² = 0.99983 | Grid frequency tracks (generation − demand) |
| `f15 = 1.10421 − 0.0041807·f07` | R² = 0.9766 | PV temperature derating, γ = **−0.418%/°C** (textbook: 0.4–0.5%/°C) |
| `f07 = f02 + 0.0219·f01 − 0.348·f04` | R² = 0.9915 | Faiman panel-temperature model (ambient + irradiance heating − wind cooling) |
| `f10 = f05·100 / (287.05·(f02+273.15))` | R² = 0.9301 | Ideal gas law, air density |
| `f18 = f04²` | R² = 0.9926 | Declared wind-speed-squared engineered feature |
| `f17 = f01·f03/100` | R² = 0.9938 | Declared irradiance×humidity engineered feature |
| `f19 = f08·f09` | R² = 0.9788 | Declared demand×storage engineered feature |
| `f11 = 0.9296 + 0.0314·f08` | R² = 0.358 | Inverter efficiency is a pure function of SOC — **not an independent quantity** |
| `f13` (wind power) | — | Turbine curve: cut-in ≈2.0 m/s, rated ≈10.75 m/s, cut-out ≈18 m/s |
| `f22` (transmission loss) | — | Rises linearly with irradiance, **saturates hard at 0.080** once G > ~900 |
| `f08, f09, f11, f19` | R² up to 0.53 | Periodic in row order — `f08` period 83,333, `f09` period 50,000 (row index encodes time) |

**Distractors confirmed dead** (ρ with target ≈ 0, verified via a null-feature control that returned ρ = −0.003):
`f11` (inverter eff.), `f08` (SOC), `f06` (cloud cover), `f20`, `f23`, `f24`, `f25`.

This matters: the problem statement names "battery-demand drawdown" as a mechanism, but
SOC itself carries essentially no signal. The drawdown term is **demand alone**, not
demand moderated by storage state.

---

## 2. What we measured about the target (via ~95 leaderboard-calibrated submissions)

With no labels, every submission's public RMSE inverts exactly to a covariance
measurement via `RMSE² = σ_y² + σ_p² + (μ_y−μ_p)² − 2·cov(y,p)`. We validated this
framework with a null-feature control before trusting it (see above), and confirmed
`σ_y ≈ 4.611` (organizer stated ≈4.6).

### 2a. Linear correlations (ρ) with the target

| feature | ρ | feature | ρ | feature | ρ |
|---|---|---|---|---|---|
| f13 (wind gen) | **+0.830** | f15 (temp-loss) | **+0.571** | f01 (irradiance) | −0.286 |
| f21 (grid freq) | **+0.828** | f10 (air density) | +0.360 | f17 (irr×hum) | −0.278 |
| f18 (wind²) | **+0.806** | f03 (humidity) | −0.050 | f12 (solar gen) | −0.251 |
| f04 (wind speed) | **+0.797** | f22 (trans. loss) | −0.109 | f19 (dem×stor) | −0.174 |
| f07 (panel temp) | **−0.568** | f09 (demand) | −0.283 | f02 (ambient T) | −0.404 |
| f11, f08, f06 | ≈0 (dead) | | | | |

**Key structural finding — suppression:** `f12` (solar generation) has *negative*
marginal correlation with the target despite almost certainly having a large positive
coefficient in the true equation. Irradiance drives generation up but also heats the
panel, and the thermal penalty dominates the raw correlation. This only resolves once
you condition on temperature.

**Key structural finding — additive, not multiplicative:** `usable = f14·f15·f11·(1−f22)`
(the "textbook" merger: gross generation × derating × inverter efficiency × transmission
efficiency) achieves ρ = 0.831 — barely better than raw wind generation alone (ρ=0.829).
If derating were multiplicative, stacking factors should lift correlation substantially.
It doesn't. **The true equation is additive**: a linear/nonlinear sum of terms, not a
product of efficiency factors.

### 2b. The dominant nonlinear structure: a thermal cutoff

By perturbing our best linear model along orthogonalized directions and reading back the
LB score, the single largest piece of unexplained variance (var explained 0.386 of the
total target variance) came from:

```
relu(f07 − 55)          gamma = -0.622   (single biggest signal measured all day)
```

A staircase of further thermal probes refined this. Reading the *shape* of the
correction (not just individual probe scores) off the accumulated model:

```
slope of correction on f07 in [25,45): +0.067/°C
slope of correction on f07 in [45,52): -0.144/°C
slope of correction on f07 in [52,58): -0.188/°C
slope of correction on f07 in [58,70): -0.319/°C
```

This is an **accelerating** slope — consistent with a **quadratic penalty above a knee**,
not a linear hinge. A grid search over `T_cell = T_amb + a·G − b·v` crossed with
`relu(T_cell − T0)^p` and `exp(T_cell/τ)` forms found the best fit at:

```
T_cell = f02 + 0.020·f01        (NOT the f07 sensor's own coefficient of 0.0219 —
                                  close, but the target uses a slightly different
                                  implied cell-temperature constant)
penalty ≈ relu(T_cell − 35)²    R² = 0.8625 against the measured correction
```

i.e. **the thermal penalty knee is at T_cell ≈ 35°C**, not at the f07 sensor reading of
55°C — the f07-based probes were catching a *noisy proxy* of the true trigger, which is
closer to a clean NOCT-style cell temperature. This was found in the last ~30 minutes of
the competition and never validated with a dedicated submission — **it is our single
best untested lead** if this were re-run.

Secondary confirmed nonlinear terms (in descending order of measured effect size):

```
relu(T_panel-44)²                          gamma = -0.316   (quadratic thermal, coarse knee)
smooth spline(f07)                         gamma = +0.305   (generalizes the thermal shape)
relu(T_panel-45)·f13                       gamma = -0.281   (thermal penalty scales with wind gen)
1[T_panel > 50]                            gamma = -0.267   (regime indicator)
relu(T_panel-60)·f13                       gamma = +0.172   (correction to the above at high T)
relu(demand - 1.0)                         gamma = -0.149   (deficit-side demand hinge)
f07 × f02 tensor (panel×ambient surface)   gamma = +0.139   (NOCT delta-T structure)
relu(f09-1.0) [amplified, deficit regime]  gamma = +0.058   (secondary deficit-regime term)
f01 spline (irradiance, standalone)        gamma = -0.110   (low-light nonlinearity)
distractors (f25, f23, f24, combined)      gamma = +0.0001  (confirms zero — sanity check)
```

The distractor probe returning γ≈0 at the very end of the search is an important
consistency check: it confirms the measurement framework never picked up spurious
signal, all the way through 95 submissions.

---

## 3. Best-guess closed-form equation

Combining the confirmed algebraic identities with the measured nonlinear structure, our
reconstruction of NDEM is:

```
T_cell = T_amb + 0.020 · Irradiance                              [cell/panel temperature]

NDEM ≈  β1 · WindGen
      + β2 · SolarGen
      − β3 · relu(T_cell − 35)²                                  [thermal derating, dominant term]
      − β4 · relu(T_cell − 35)² · WindGen                        [derating scales with generation]
      − β5 · Demand
      − β6 · relu(Demand − 1.0)                                  [deficit-regime kink]
      − β7 · TxLoss · GrossGen                                   [transmission loss, saturating ~0.08]
      + c
```

with the constants β1..β7 unrecovered exactly — our measurement only pins their
*direction and rough relative magnitude*, not tight values, because we spent submissions
on many small orthogonal probes rather than a small number of coefficient fits against a
committed parametric form.

**This is consistent with, but less precise than, what the winning teams almost
certainly did**: read the data dictionary (which literally names every column's physical
role), write down the standard published forms for PV temperature derating, wind power
curves, and net-load/dispatch margin, and calibrate a handful of global coefficients
against 5–15 submissions. DP Team's 5 submissions and near-zero implied noise floor
(≈0.10, from ENERGion's 0.104) are not reachable by leaderboard measurement in any
practical number of tries — they require having the correct functional form *before*
the first submission, not discovering it via 90 measurements after.

---

## 4. Where we went wrong

1. **Switched from deriving to measuring too early.** Our first physics-derived model
   (the multiplicative "usable energy" form) scored 2.76. Instead of concluding the
   *functional form* was wrong and searching harder for the right one (additive,
   thermal-cutoff-based), we pivoted to leaderboard-measurement, which is slower and has
   a much lower ceiling.
2. **Measurement finds structure but not exact coefficients.** Each orthogonal probe
   costs one submission and buys ~0.02–0.03 RMSE. Closing a gap of ~0.47 (our final
   error above the noise floor) this way needs dozens of batches — hundreds of
   submissions — not the ~25 we had.
3. **The thermal knee was mis-located for most of the day.** We anchored on f07=55°C
   early (from the first strong probe) and kept refining around that anchor. The
   better-fitting T_cell=35°C form was only found in the last half hour, via a residual
   regression the search hadn't tried earlier, and was never submitted to confirm.
4. **Time management in the final hours.** We held all reserved submissions per
   instruction to defend rank, which meant the last hour was a scramble rather than a
   considered attempt at the parametric form above.

## 5. If this were re-run

The highest-leverage next step, cheaply testable: submit the T_cell=35°C quadratic form
directly (β3, β4 above) rather than continuing to add small orthogonal probes. It was
never tested standalone. Everything else in section 1–2 is solid ground to start a
proper closed-form fit from turn one, rather than rediscovering it under time pressure.
