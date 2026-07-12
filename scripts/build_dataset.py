"""Build the California ZIP-year housing panel.

One row = one California ZIP code observed in one base year, with the realized five-year
home-price growth that followed. See Project/DATASET_DESIGN.md for the full specification.

    python scripts/build_dataset.py --base-years 2020            # Lab 1 snapshot
    python scripts/build_dataset.py --base-years 2016 2017 2018 2019 2020   # D1 panel

Requires ACS files from scripts/fetch_acs.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

ZHVI = ROOT / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
ZORI = ROOT / "Zip_zori_uc_sfrcondomfr_sm_month.csv"
FRED = RAW / "MORTGAGE30US.csv"

LISTINGS = {
    "for_sale_inventory": "Zip_invt_fs_uc_sfrcondo_sm_month.csv",
    "new_listings": "Zip_new_listings_uc_sfrcondo_sm_month.csv",
    "price_cut_pct": "Zip_perc_listings_price_cut_uc_sfrcondo_sm_month.csv",
    "median_list_price": "Zip_mlp_uc_sfrcondo_sm_month.csv",
}
# all listings series begin 2018-03; earlier years are structurally absent
LISTINGS_START_YEAR = 2018


def dec(year: int) -> str:
    return f"{year}-12-31"


def load_zillow_ca(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"RegionName": str})
    df = df[df["State"] == "CA"].copy()
    df["RegionName"] = df["RegionName"].str.zfill(5)
    return df.set_index("RegionName")


def mortgage_rate_by_year() -> pd.Series:
    """Annual mean of the weekly national 30-year fixed rate."""
    f = pd.read_csv(FRED, parse_dates=["observation_date"])
    f = f.rename(columns={"observation_date": "date", "MORTGAGE30US": "rate"})
    return f.groupby(f["date"].dt.year)["rate"].mean()


def load_acs(year: int) -> pd.DataFrame:
    path = RAW / "acs" / f"acs5_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: CENSUS_API_KEY=... python scripts/fetch_acs.py"
        )
    return pd.read_csv(path, dtype={"zip_code": str}).set_index("zip_code")


def build_base_year(year: int, zhvi, zori, listings, rates) -> pd.DataFrame:
    """Assemble every California ZIP for a single base year."""
    t, prev, t3, fut = dec(year), dec(year - 1), dec(year - 3), dec(year + 5)

    for col, label in [(t, "base"), (prev, "prior"), (fut, "outcome")]:
        if col not in zhvi.columns:
            raise ValueError(
                f"base_year {year}: ZHVI missing {label} month {col}")

    # A row only exists if we can compute both the base value and the realized outcome.
    keep = zhvi[t].notna() & zhvi[prev].notna() & zhvi[fut].notna()
    z = zhvi[keep]

    df = pd.DataFrame(index=z.index)
    df.index.name = "zip_code"
    df["base_year"] = year
    df["city"] = z["City"]
    df["county"] = z["CountyName"]

    df["current_home_value"] = z[t]
    df["future_5yr_home_value"] = z[fut]
    df["home_value_1yr_growth_pct"] = (z[t] / z[prev] - 1) * 100
    df["home_value_3yr_growth_pct"] = (
        (z[t] / z[t3] - 1) * 100 if t3 in z.columns else np.nan
    )
    df["future_5yr_growth_pct"] = (z[fut] / z[t] - 1) * 100

    # ZORI: supplementary only. Coverage is 28% in 2020 and 15% in 2016, and the missingness
    # is not random (it skews to large, liquid rental markets), so it is never imputed and
    # never used as a core model feature. ACS median gross rent fills that role instead.
    df["zori_rent"] = zori[t].reindex(
        df.index) if t in zori.columns else np.nan
    df["zori_rent_available"] = df["zori_rent"].notna().astype(int)

    # ACS: level at t, population at t-5 for non-overlapping growth.
    acs = load_acs(year).reindex(df.index)
    acs_lag = load_acs(year - 5).reindex(df.index)

    df["median_household_income"] = acs["B19013_001E"]
    df["median_gross_rent"] = acs["B25064_001E"]
    df["population"] = acs["B01003_001E"]
    df["housing_units"] = acs["B25001_001E"]

    labor_force = acs["B23025_003E"].replace(0, np.nan)
    df["unemployment_rate"] = acs["B23025_005E"] / labor_force * 100

    adults = acs["B15003_001E"].replace(0, np.nan)
    bachelor_plus = acs[["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]].sum(
        axis=1, min_count=1
    )
    df["bachelor_degree_pct"] = bachelor_plus / adults * 100

    pop_lag = acs_lag["B01003_001E"].replace(0, np.nan)
    df["population_growth_pct"] = (acs["B01003_001E"] / pop_lag - 1) * 100

    df["mortgage_rate"] = rates.get(year, np.nan)

    # Affordability ratios.
    df["price_to_income_ratio"] = df["current_home_value"] / \
        df["median_household_income"].replace(0, np.nan)
    df["price_to_rent_ratio"] = df["current_home_value"] / \
        (df["median_gross_rent"].replace(0, np.nan) * 12)

    # Listings block: 2018+ only. Counts are normalized by housing units, otherwise they
    # merely proxy for how large the ZIP code is.
    per_1k = df["housing_units"].replace(0, np.nan) / 1000
    if year >= LISTINGS_START_YEAR:
        for name, lf in listings.items():
            col = lf[t].reindex(df.index) if t in lf.columns else np.nan
            df[name] = col
        df["inventory_per_1k_housing_units"] = df["for_sale_inventory"] / per_1k
        df["new_listings_per_1k_housing_units"] = df["new_listings"] / per_1k
        df["list_to_value_ratio"] = df["median_list_price"] / \
            df["current_home_value"]
    else:
        for name in ["for_sale_inventory", "new_listings", "price_cut_pct", "median_list_price",
                     "inventory_per_1k_housing_units", "new_listings_per_1k_housing_units",
                     "list_to_value_ratio"]:
            df[name] = np.nan

    return df.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-years", nargs="+", type=int,
                    default=[2016, 2017, 2018, 2019, 2020])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)

    zhvi = load_zillow_ca(ZHVI)
    zori = load_zillow_ca(ZORI)
    listings = {k: load_zillow_ca(RAW / v) for k, v in LISTINGS.items()}
    rates = mortgage_rate_by_year()

    frames = [build_base_year(y, zhvi, zori, listings, rates)
              for y in sorted(args.base_years)]
    panel = pd.concat(frames, ignore_index=True)

    dupes = panel.duplicated(subset=["zip_code", "base_year"]).sum()
    if dupes:
        raise AssertionError(
            f"{dupes} duplicate (zip_code, base_year) rows -- join bug")

    # Impossible values -> NaN, so they are handled as missing rather than silently modeled.
    panel.loc[panel["current_home_value"] <= 0, "current_home_value"] = np.nan
    panel.loc[panel["median_household_income"]
              <= 0, "median_household_income"] = np.nan
    panel.loc[panel["unemployment_rate"] > 100, "unemployment_rate"] = np.nan

    if args.out:
        out = PROC / args.out
    elif len(args.base_years) == 1:
        out = PROC / f"ca_zip_housing_{args.base_years[0]}_snapshot.csv"
    else:
        out = PROC / "ca_zip_housing_panel.csv"
    panel.to_csv(out, index=False)

    print(f"\nWrote {out.relative_to(ROOT)}")
    print(f"  rows: {len(panel)}   columns: {panel.shape[1]}")
    print(
        f"  rows per base year:\n{panel['base_year'].value_counts().sort_index().to_string()}")
    print("\n  missing values by column:")
    miss = panel.isna().sum()
    for c, n in miss[miss > 0].sort_values(ascending=False).items():
        print(f"    {c:<36} {n:>6}  ({n / len(panel) * 100:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
