"""
Data Cleaning & Processing
============================
WHAT THIS FILE DOES:
  Takes the raw scraped CSV from Part 1 (01_data_collection/output/
  scraped_cutoffs.csv) and turns it into one clean, analysis-ready table:
  02_data_processing/output/kcet_cleaned.csv

  This is the file Part 3 (model training) will actually load - it never
  touches the raw scraped data directly. That separation matters: if the
  raw data has quirks (typos in college names, missing ranks, duplicate
  rows), you fix them ONCE here, and every downstream step benefits.

WHAT "CLEANING" MEANS HERE, SPECIFICALLY:
  1. Type conversion       - closing_rank, year come in as text from CSV/
                              PDF extraction; convert to actual numbers
  2. Missing values         - PDF parsing sometimes leaves blank cells
                              (merged table cells, OCR gaps) - drop or flag
  3. Duplicate rows         - same (year, round, college, branch, category)
                              appearing twice, usually from PDF parsing
                              picking up a repeated header row
  4. Name standardization   - "RVCE" vs "R V College of Engineering" vs
                              "RV College of Engg" all need to become one
                              consistent value, or grouping/merging breaks
  5. Category standardization - KEA sometimes uses "GM" and sometimes
                              "General Merit" - normalize to one code set
  6. Outlier sanity checks  - a closing rank of 0 or negative, or a rank
                              higher than the total number of candidates,
                              is a parsing error, not real data - flag it
  7. Round ordering         - "Final" needs to sort AFTER numbered rounds,
                              not alphabetically before them

HOW TO RUN:
  python clean_data.py
  -> reads  ../01_data_collection/output/scraped_cutoffs.csv
  -> writes output/kcet_cleaned.csv
  -> writes output/cleaning_report.txt (what was fixed/dropped, and why)
"""

import os
import pandas as pd
import numpy as np

INPUT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "01_data_collection", "output", "scraped_cutoffs.csv"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "kcet_cleaned.csv")
REPORT_TXT = os.path.join(OUTPUT_DIR, "cleaning_report.txt")

# Maps messy/alternate college name spellings to one canonical name.
# Extend this as you find more variants in real scraped data - this is
# the single most common way government PDF data breaks downstream
# grouping/filtering (the same college silently becoming 3 "different"
# colleges because of spelling differences).
COLLEGE_NAME_ALIASES = {
    "rv college of engineering": "RV College of Engineering",
    "r v college of engineering": "RV College of Engineering",
    "rvce": "RV College of Engineering",
    "bms college of engineering": "BMS College of Engineering",
    "b m s college of engineering": "BMS College of Engineering",
    "bmsce": "BMS College of Engineering",
    "ms ramaiah institute of technology": "MS Ramaiah Institute of Technology",
    "msrit": "MS Ramaiah Institute of Technology",
    "ramaiah institute of technology": "MS Ramaiah Institute of Technology",
}

# Maps messy category labels to standardized KEA category codes.
CATEGORY_ALIASES = {
    "general merit": "GM",
    "gm": "GM",
    "general": "GM",
    "sc": "SC",
    "scheduled caste": "SC",
    "st": "ST",
    "scheduled tribe": "ST",
    "category 1": "CAT1",
    "cat-1": "CAT1",
    "cat1": "CAT1",
}

# Round display order (for sorting) - "Final" is not alphabetically last,
# so a plain string sort would put it before "Round 2".
ROUND_ORDER = {"1": 0, "2": 1, "3": 2, "Final": 99}


def load_raw(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data not found at {path}\n"
            "Run 01_data_collection/scraper.py or generate_sample_data.py first."
        )
    return pd.read_csv(path)


def standardize_text_column(series, alias_map):
    """Lowercases + strips for matching against alias_map, but keeps the
    ORIGINAL value (title-cased as fallback) for anything not in the map,
    so unrecognized values aren't silently destroyed - just left as-is."""
    def normalize(value):
        if pd.isna(value):
            return value
        key = str(value).strip().lower()
        return alias_map.get(key, str(value).strip())
    return series.apply(normalize)


def clean(df):
    report_lines = []
    start_rows = len(df)
    report_lines.append(f"Starting rows: {start_rows}")

    # --- 1. Type conversion ---
    df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # --- 2. Missing values ---
    before = len(df)
    missing_rank = df["closing_rank"].isna().sum()
    missing_year = df["year"].isna().sum()
    df = df.dropna(subset=["closing_rank", "year", "college_name", "branch", "category"])
    report_lines.append(
        f"Dropped {before - len(df)} rows with missing required fields "
        f"(missing closing_rank: {missing_rank}, missing year: {missing_year})"
    )

    # --- 3. Outlier sanity checks ---
    before = len(df)
    df = df[(df["closing_rank"] > 0) & (df["closing_rank"] < 300000)]
    report_lines.append(
        f"Dropped {before - len(df)} rows with impossible closing_rank "
        f"(<=0 or >=300000 - KCET candidate pool doesn't exceed this)"
    )

    # --- 4. Name standardization ---
    df["college_name"] = standardize_text_column(df["college_name"], COLLEGE_NAME_ALIASES)

    # --- 5. Category standardization ---
    df["category"] = standardize_text_column(df["category"], CATEGORY_ALIASES)

    # --- 6. Round ordering (for correct chronological sorting later) ---
    df["round"] = df["round"].astype(str)
    df["round_sort_order"] = df["round"].map(ROUND_ORDER).fillna(50)

    # --- 7. Duplicate rows ---
    before = len(df)
    dedup_keys = ["year", "round", "college_code", "branch", "category"]
    df = df.sort_values("closing_rank").drop_duplicates(subset=dedup_keys, keep="first")
    report_lines.append(
        f"Dropped {before - len(df)} duplicate rows "
        f"(same year/round/college/branch/category appearing more than once)"
    )

    # --- Final type cleanup ---
    df["year"] = df["year"].astype(int)
    df["closing_rank"] = df["closing_rank"].astype(int)
    df = df.sort_values(["college_name", "branch", "category", "year", "round_sort_order"])
    df = df.reset_index(drop=True)

    report_lines.append(f"Final rows: {len(df)}")
    report_lines.append(f"Rows dropped overall: {start_rows - len(df)} "
                         f"({(start_rows - len(df)) / start_rows * 100:.1f}%)")
    report_lines.append(f"Unique colleges: {df['college_name'].nunique()}")
    report_lines.append(f"Unique branches: {df['branch'].nunique()}")
    report_lines.append(f"Years covered: {sorted(df['year'].unique().tolist())}")

    return df, report_lines


def main():
    print("Loading raw data...")
    df = load_raw(INPUT_CSV)
    print(f"  Loaded {len(df)} raw rows.")

    print("Cleaning...")
    cleaned, report_lines = clean(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cleaned.to_csv(OUTPUT_CSV, index=False)
    with open(REPORT_TXT, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nDone. Saved {len(cleaned)} clean rows -> {OUTPUT_CSV}")
    print(f"Cleaning report -> {REPORT_TXT}")
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
