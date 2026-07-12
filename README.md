# MSCS 634 — Lab 1

## Factors Associated with Five-Year Home Price Growth in California

**Name:** Sammy Sheu
**Course:** MSCS 634 — Big Data and Data Mining
**Assignment:** Lab 1 — Data Visualization, Preprocessing, and Statistical Analysis

---

## Purpose

This lab applies data visualization, preprocessing, and statistical analysis to a dataset of
**1,528 California ZIP codes observed in 2020**, in order to investigate what was associated with
**home-price growth over the following five years (December 2020 → December 2025)**.

Because that five-year window has fully elapsed, the target is a **measured outcome, not a
forecast** — the analysis can be validated against what actually happened.

The lab is deliberately built as a **single-year slice of the course Project's dataset**, so the
cleaning decisions, discretized categories, and correlation findings carry forward directly into
Project Deliverables 1–4 rather than being thrown away.

## Dataset

| | |
|---|---|
| **Unit of analysis** | One California ZIP code, base year 2020 |
| **Records** | 1,528 |
| **Attributes** | 28 raw → 11 core model features |
| **Target** | `future_5yr_growth_pct` — % change in home value, Dec 2020 → Dec 2025 |

### Sources

| Source | Contributes |
|---|---|
| **Zillow ZHVI** (ZIP, smoothed, seasonally adjusted) | Home values: 2020 level, 1- and 3-year momentum, 2025 outcome |
| **Zillow ZORI** (ZIP) | Observed asking rent — sparse; used to demonstrate missing-value handling |
| **US Census ACS 5-Year** (ZCTA) | Median household income, median gross rent, population, unemployment, education |
| **FRED `MORTGAGE30US`** | National 30-year fixed mortgage rate (2020 annual mean) |

Reproduce the dataset with:

```bash
pip install -r requirements.txt
export CENSUS_API_KEY=your_key        # https://api.census.gov/data/key_signup.html
python scripts/fetch_acs.py
python scripts/build_dataset.py --base-years 2020
```

---

## Key insights

> Correlations from Step 2 are computed on `df_raw` (1,528 rows); those from Step 4 are computed on
> `reduced_df` (1,485 rows, after outlier removal). The source of each figure is noted below.

### 1. How prosperous a market is tells you almost nothing about how fast it will grow

The single most important result. Median household income predicts a ZIP code's home **value**
strongly (**r = 0.738**) — but `current_home_value` correlates with subsequent five-year **growth**
at **r = −0.020**, indistinguishable from zero. Education is no better (**r = −0.042**).

**Prosperity in 2020 and appreciation over 2020–2025 are essentially independent questions.**
San Francisco proves it: the state's richest and best-educated market *declined* **7.4%**.

This validates the Project's core design decision — predicting *growth* rather than *future price*.
Predicting price levels would have let `current_home_value` dominate the model and produce a
flattering R² that concealed having learned nothing about growth.

### 2. Momentum and urban scale lead — and both are still weak

The two strongest correlates are `home_value_1yr_growth_pct` (**r = +0.312**) and `population`
(**r = +0.277**): markets already appreciating kept appreciating, and larger, more urban ZIP codes
appreciated faster. But r ≈ 0.31 explains only ~10% of the variance. **No single feature comes close
to explaining five-year growth.** Deliverable 2's regression must rely on feature *combinations*, and
expectations for accuracy should be correspondingly modest.

`population` also carries a methodological lesson — see the note on dimension elimination below.

### 3. The affordability-constraint hypothesis holds directionally, but weakly

`price_to_income_ratio` is the strongest negative correlate (**r = −0.173** after outlier removal;
**−0.214** on the raw data) — already-stretched markets grew more slowly. The direction supports the
hypothesis; the magnitude makes affordability a **secondary** factor, not a primary driver.

### 4. Growth is a geographic story more than a demographic one

| Highest growth | | Lowest growth | |
|---|---|---|---|
| Orange County | +49.0% | Mendocino County | −16.4% |
| Santa Barbara County | +46.0% | Humboldt County | −14.4% |
| San Diego County | +40.5% | San Francisco County | −7.4% |

Southern California's coast sharply outperformed the rural North Coast. None of the current features
capture coastal proximity, land scarcity, or remote-work exposure — so a substantial share of the
variance is driven by factors **the dataset does not yet measure.**

### 5. A national feature is statistically inert in a single-year snapshot

`mortgage_rate` has **zero variance** here (every row is 2020's 3.11%), so its correlation with the
target is **undefined**. It cannot explain differences *between* ZIP codes. It only becomes
informative in the Project's multi-year panel, where it varies *across* base years — a clean
illustration of how a feature can be economically important yet statistically useless given the
wrong unit of analysis.

### Statistical summary

| Measure | Five-year growth | 2020 home value |
|---|---|---|
| Mean | 24.3% | $705,351 |
| Median | 26.2% | $580,468 |
| Std. deviation | 17.3 pts | — |
| Range | −45.5% → +127.8% | up to $6.48M |
| Skew | −0.29 (mild left) | +3.01 (strong right) |

Growth and value are skewed in **opposite** directions: the growth distribution has a long tail of
sharp *decliners*, while home values have a long tail of ultra-*expensive* markets. The right skew in
home values is why the **median** — not the mean — was used for imputation.

---

## Preprocessing decisions

**Missing values.** Handled per column rather than with a blanket fill — a ZIP code with zero income
does not exist, and filling with `0` would drag every mean and correlation toward it.

| Columns | Action | Reason |
|---|---|---|
| `zip_code`, `current_home_value`, `future_5yr_growth_pct` | Drop row | Never impute the target or its denominator — that fabricates the answer |
| ACS features (0.5–6.9% missing) | Median imputation | Modest missingness; median is robust to the heavy right skew |
| `zori_rent` (**71.9% missing**) | Retain, do not impute | Not missing at random — see below |

**Outliers.** The IQR rule on `future_5yr_growth_pct` flags **43 ZIP codes (2.8%)** — 30 below the
lower bound, 13 above. Inspection showed all 43 are **legitimate markets, not errors**: the low
outliers are rural North Coast declines (Bridgeville, Redway, Laytonville, −35% to −45%), the high
ones are affluent SoCal coastal booms (Carlsbad +127.8%, Irvine, Newport Beach). Removal is
*demonstrated* here to satisfy the lab, but `df_raw`, `df_clean`, and `df_no_outliers` are kept side
by side and the original is never overwritten — the Project will **flag** rather than delete, because
discarding them would remove exactly the behaviour the model needs to explain.

**Data reduction.** 30% sampling (demonstration only — the Project uses the full dataset) plus
dimension elimination: `future_5yr_home_value` (the target restated → leakage), `zori_rent` /
`zori_rent_available` (see above), and `housing_units` (only needed to normalize the listings
features, which Lab 1 does not use).

> **Why `population` was kept — the real lesson of dimension elimination.** An earlier version of
> this analysis dropped `population`, reasoning that raw size is "a size proxy, not a growth signal."
> **The correlation table refutes that:** `population` is the *second-strongest* correlate with
> five-year growth in the entire dataset (**r = +0.277**). Dropping it would have discarded genuine
> signal, on a justification the notebook's own output contradicts a few cells later. Columns must be
> eliminated on **evidence**, not on intuition about what ought to matter.

**Scaling and discretization.** Min-Max scaling onto `[0, 1]` — necessary because
`current_home_value` (~10⁶) would otherwise dominate `unemployment_rate` (~10⁰) in any
distance-based algorithm. Growth was discretized into Low/Medium/High terciles; that
`growth_category` is reused as the **classification label and association-rule item in Deliverable 3.**

---

## Challenges

1. **Zillow's rent index (ZORI) proved unusable as a feature.** It covers only 431 of 1,528
   California ZIP codes (**71.9% missing**), and the missingness is *not random* — coverage skews
   toward large, liquid rental markets. Imputing nearly three-quarters of a column would have
   invented data and biased it toward big cities. **Resolved** by switching to ACS median gross rent
   (6.9% missing) as the rent variable, keeping ZORI only as the missing-value demonstration.

2. **The Census API rejected an unactivated key**, returning an `Invalid Key` page despite a
   correctly formatted 40-character key. **Resolved** by activating it via the link the Census Bureau
   emails separately.

3. **ZIP codes are not ZCTAs.** Zillow uses postal ZIP codes; the Census publishes ZIP Code
   Tabulation Areas — similar but not identical geographies. **Documented as a limitation** rather
   than papered over.

4. **Whether to delete outliers.** See above — inspection showed they were real markets, so the
   decision was to demonstrate removal but preserve the originals.

---

## Repository structure

```
Lab1/
├── notebooks/
│   └── MSCS_634_Lab_1.ipynb          # all four lab steps, executed with outputs
├── data/processed/
│   └── ca_zip_housing_2020_snapshot.csv
├── screenshots/                       # required screenshots (11 marked in the notebook)
└── README.md
```

Screenshot points are marked in the notebook with **📸 SCREENSHOT n** headers.

---

## A note on causal language

This is **observational** data. Every relationship reported above is an **association**, not a
demonstrated cause. *"Price-to-income ratio is negatively associated with five-year growth"* is
supportable; *"high prices cause slower growth"* is not. No correlation coefficient, and no model
accuracy score, establishes causation here.
