# Recovered Feature Dictionary

The competition supplies `data_dictionary.csv` naming each column's physical role. This
document records what was **recovered from the data itself** — the exact algebraic
relationships the generator used, obtained by regression across all 500,000 rows with no
target information.

Every relationship below is independently verifiable by running
[`src/01_discovery/eda02_relations.py`](../src/01_discovery/eda02_relations.py).

---

## Column reference

| Column | Declared role | Recovered nature |
|---|---|---|
| `feature_01` | Solar irradiance proxy | Independent draw, ~0–1275 W/m² |
| `feature_02` | Ambient temperature proxy | Independent, Gaussian μ≈27 °C, σ≈7 |
| `feature_03` | Relative humidity proxy | Independent, μ≈66.9 %, σ≈16.9 |
| `feature_04` | Wind speed proxy | Independent, Weibull-like, μ≈5.33 m/s |
| `feature_05` | Atmospheric pressure proxy | Independent, Gaussian μ≈1013 hPa |
| `feature_06` | Cloud-cover proxy | Independent, ~Beta on [0,1] |
| `feature_07` | Panel/system temperature | **Derived** — Faiman model (below) |
| `feature_08` | Battery state-of-charge | **Periodic** — sinusoid, period 83,333 rows |
| `feature_09` | Normalized system demand | **Periodic** — sinusoid, period 50,000 rows |
| `feature_10` | Air-density proxy | **Derived** — ideal gas law |
| `feature_11` | Inverter efficiency proxy | **Derived** — pure function of SOC |
| `feature_12` | Solar generation proxy | **Derived** — concave in irradiance |
| `feature_13` | Wind generation proxy | **Derived** — turbine power curve |
| `feature_14` | Gross generation proxy | **Derived** — exactly `f12 + f13` |
| `feature_15` | Temperature-loss factor | **Derived** — PV temperature coefficient |
| `feature_16` | Precipitation proxy | ~91% noise (only R²=0.09 from cloud) |
| `feature_17` | Irradiance–humidity interaction | **Derived** — `f01·f03/100` |
| `feature_18` | Wind-speed-square proxy | **Derived** — `f04²` |
| `feature_19` | Demand–storage interaction | **Derived** — `f08·f09` |
| `feature_20` | Synthetic noise sensor | Distractor — confirmed ρ≈0 with target |
| `feature_21` | Grid frequency proxy | **Derived** — generation/demand imbalance |
| `feature_22` | Transmission-loss proxy | **Derived** — clipped function of irradiance |
| `feature_23` | Unrelated economic signal | Distractor — confirmed ρ≈0 |
| `feature_24` | Slow calibration drift | Distractor, but **periodic** (period 250,000) |
| `feature_25` | Synthetic noise sensor | Distractor — confirmed ρ≈0 |

---

## Exact recovered identities

### Generation balance
```
f14 = f12 + f13                                              R² = 0.99992
```
Gross generation is exactly solar plus wind. Residual 0.0496 — the tightest relationship
in the dataset, implying all three were computed from clean latent values.

### Grid frequency tracks supply–demand imbalance
```
f21 = 49.99 + 0.900·f14 − 0.880·f09                          R² = 0.99983
```
Equivalently `f21 ≈ 50 + 0.9·(gross generation − demand)`. Physically correct: grid
frequency rises above nominal when generation exceeds load. Provides a redundant, low-
noise route to `f14` when it is missing.

### PV temperature derating
```
f15 = 1.10421 − 0.0041807·f07                                R² = 0.9766
```
Rearranges to `f15 = 1 + γ·(T_panel − 25)` with **γ = −0.418 %/°C** — squarely inside the
published 0.4–0.5 %/°C range for crystalline silicon, and referenced to the 25 °C STC
condition.

### Panel temperature (Faiman model with wind cooling)
```
f07 = f02 + 0.0219·f01 − 0.348·f04                           R² = 0.9915
```
Ambient temperature, plus irradiance heating, minus wind convective cooling. The
irradiance coefficient 0.0219 corresponds to `(NOCT−20)/800` with NOCT ≈ 37.5 °C.

### Air density (ideal gas law)
```
f10 = f05·100 / (287.05·(f02 + 273.15))                      R² = 0.9301
```
`R = 287.05 J/(kg·K)` is the exact specific gas constant for dry air.

### Declared engineered interactions
```
f17 = f01·f03/100                                            R² = 0.9938
f18 = f04²                                                   R² = 0.9926
f19 = f08·f09                                                R² = 0.9788
```

### Inverter efficiency is a function of storage state
```
f11 = 0.9296 + 0.0314·f08                                    R² = 0.358
```
Not an independent quantity. Combined with its measured ρ ≈ 0 against the target, this
whole branch is irrelevant to `NDEM` despite the "Conversion" label.

### Wind turbine power curve
`f13` follows a standard piecewise power curve rather than pure `v³`:
```
cut-in  ≈ 2.0 m/s
rated   ≈ 10.75 m/s
cut-out ≈ 18 m/s
```
Fitting this curve gives residual 0.49 versus 2.35 for an uncapped cubic.

### Transmission loss saturates in irradiance
`f22` rises linearly with irradiance then **clips hard at 0.080**:
```
G:    75    225    375    525    675    825    975   1125   1275
f22: .0357 .0444 .0544 .0647 .0743 .0795 .0800 .0800 .0800
```
Within-bin standard deviation collapses from 0.0067 to 0.0030 at the cap, confirming a
hard clip rather than a smooth asymptote. Approximately
`f22 ≈ min(0.080, 0.0332 + 4.98e-5·G)`.

---

## Temporal structure

`row_id` is not an index — it encodes time. Four features are exact single-harmonic
sinusoids in row order:

| feature | period (rows) | R² (1 harmonic) |
|---|---|---|
| `f08` battery SOC | 83,333 | 0.5276 |
| `f09` demand | 50,000 | 0.2619 |
| `f11` inverter efficiency | 83,333 | 0.1902 |
| `f24` calibration drift | 250,000 | 0.3103 |

Adding harmonics beyond the first does not improve the fit, confirming pure sinusoids.
All 21 remaining features show phase-R² < 0.001 — drawn i.i.d.

Verified against a shuffle control: real `f08` autocorrelation 0.527 at lag 5,000;
shuffled 0.000.

---

## Missingness structure

Overall ~26% of cells absent, in two tiers:

- **~24.9%** baseline (random sensor outage) for most columns
- **~30.4%** for `f07`, `f11`, `f15`, `f21` — the condition-dependent failure group

Missingness correlates weakly with physical conditions (max |corr| 0.041 against a fitted
model), confirming the documented condition-dependent mechanism. This is exploited by
supplying the missingness mask to the imputer rather than by direct modelling.
