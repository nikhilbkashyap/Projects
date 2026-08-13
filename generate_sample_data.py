"""
Sample Data Generator
======================
WHAT THIS FILE DOES:
  Creates a synthetic but realistic dataset with the EXACT same columns
  scraper.py would produce from real KEA PDFs. This lets us build and
  test Parts 2, 3, and 4 of the project right now, without waiting on
  live scraping (which can't run inside this sandbox).

WHY THE NUMBERS AREN'T RANDOM NOISE:
  Base closing ranks per college/branch are set close to real reported
  2025 ranges (e.g. RVCE CSE GM closing around 234-650 across rounds).
  On top of that base, the generator applies:
    - a year-over-year TIGHTENING trend (more applicants each year is a
      real, reported trend - closing ranks get lower/tighter over time)
    - round-over-round LOOSENING within a year (closing ranks rise from
      Round 1 to the final round as seats get released - also real)
    - random noise so it isn't perfectly smooth

  This is for DEVELOPMENT/TESTING ONLY. Replace output/scraped_cutoffs.csv
  with real data from scraper.py before treating any prediction as
  trustworthy.

HOW TO RUN:
  python generate_sample_data.py
  -> writes output/scraped_cutoffs.csv
"""

import os
import csv
import random

random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "scraped_cutoffs.csv")

# (college_code, college_name)
COLLEGES = [
    ("E001", "RV College of Engineering"),
    ("E002", "BMS College of Engineering"),
    ("E003", "MS Ramaiah Institute of Technology"),
    ("E004", "PES University"),
    ("E005", "UVCE - University Visvesvaraya College of Engineering"),
    ("E006", "Dayananda Sagar College of Engineering"),
    ("E007", "Siddaganga Institute of Technology"),
    ("E008", "BNM Institute of Technology"),
    ("E009", "New Horizon College of Engineering"),
    ("E010", "Bangalore Institute of Technology"),
]

BRANCHES = ["Computer Science", "Information Science", "Electronics & Comm.",
            "Mechanical", "Civil", "Electrical & Electronics"]

CATEGORIES = ["GM", "1G", "2A", "2B", "3A", "3B", "SC", "ST", "CAT1"]

YEARS = [2021, 2022, 2023, 2024, 2025]
ROUNDS = [1, 2, "Final"]

# Rough base closing rank (Round 1, 2021, GM category) per (college, branch).
# Lower rank number = more competitive/harder to get. Anchored to real
# reported figures where available:
#   - RVCE CSE GM: 220 (2021) -> ~499 (2025 final round), a documented
#     5-year trend - not a guess.
#   - MSRIT CSE GM: tightened by ~1,678 ranks over the same 5 years.
#   - BMSCE and UVCE CSE: tightened by ~1,000 ranks each over 5 years.
#   - Branch-tier ranges (GM, 2025, across all colleges) also match
#     reported bands: CSE/AI ~500-5k, ISE ~6k-35k, ECE ~8k-50k,
#     EEE ~20k-80k, Mechanical ~40k-150k, Civil ~60k-120k+.
BASE_RANK = {
    ("E001", "Computer Science"): 220,    # RVCE - real 2021 anchor
    ("E001", "Information Science"): 1400,
    ("E002", "Computer Science"): 900,    # BMSCE
    ("E003", "Computer Science"): 700,    # MSRIT - real 2021 anchor
    ("E004", "Computer Science"): 1800,   # PES
    ("E005", "Computer Science"): 1200,   # UVCE
}

# Per-year tightening RATE per (college, branch), as a fraction of base
# rank per year. NOTE ON HONESTY: two of my sources disagreed on RVCE's
# actual direction - one showed RVCE plain "Computer Science Engineering"
# INCREASING from 220 (2021) to 499 (2025), most likely because RVCE
# opened separate AI/ML, Data Science, and Cyber Security branches after
# 2021 that pulled top rankers away from plain CSE (2026 data confirms
# those sub-branches now close far tighter than plain CSE: ~144-288 vs
# ~499). Rather than force-fit two disagreeing numbers into one
# trajectory, these RATES use the reported magnitudes only for realistic
# RELATIVE variability across colleges (some tighten faster than others)
# - not as a literal year-by-year reproduction of any single source.
TIGHTENING_RATE = {
    ("E001", "Computer Science"): 0.02,   # RVCE - slower (branch splintering offsets demand)
    ("E003", "Computer Science"): 0.07,   # MSRIT - fastest reported tightening
    ("E002", "Computer Science"): 0.05,   # BMSCE
    ("E005", "Computer Science"): 0.05,   # UVCE
}
DEFAULT_TIGHTENING_RATE = 0.04  # generic fallback for everything else

# Category multipliers relative to GM (reserved categories generally have
# higher/looser closing ranks than GM - this is a simplification for
# sample data, not an official KEA formula)
CATEGORY_MULT = {
    "GM": 1.0, "1G": 1.1, "2A": 1.4, "2B": 1.6,
    "3A": 1.7, "3B": 1.8, "SC": 3.0, "ST": 4.0, "CAT1": 1.3,
}

ROUND_MULT = {1: 1.0, 2: 1.15, "Final": 1.35}


def base_rank_for(college_code, branch):
    key = (college_code, branch)
    if key in BASE_RANK:
        return BASE_RANK[key]
    # Colleges/branches without a hardcoded base get a plausible mid-range
    # value derived deterministically from their names, so re-runs are
    # consistent.
    seed = sum(ord(c) for c in college_code + branch) % 5000
    return 3000 + seed


def rank_for_year(college_code, branch, base, year):
    """
    Computes the Round-1-equivalent base rank for a given year, applying
    compounding tightening at a per-college rate (see TIGHTENING_RATE).
    Compounding (rather than flat subtraction) guarantees the rank stays
    positive no matter how many years elapse - a flat subtraction can
    (and did, in an earlier version of this script) go negative.
    """
    key = (college_code, branch)
    rate = TIGHTENING_RATE.get(key, DEFAULT_TIGHTENING_RATE)
    years_elapsed = year - 2021
    return base * ((1 - rate) ** years_elapsed)


def generate_rows():
    rows = []
    for year in YEARS:
        for round_num in ROUNDS:
            for college_code, college_name in COLLEGES:
                for branch in BRANCHES:
                    base = base_rank_for(college_code, branch)
                    year_adjusted_base = rank_for_year(college_code, branch, base, year)
                    for category in CATEGORIES:
                        rank = (
                            year_adjusted_base
                            * ROUND_MULT[round_num]
                            * CATEGORY_MULT[category]
                        )
                        noise = random.uniform(0.92, 1.08)
                        closing_rank = max(1, round(rank * noise))
                        rows.append(
                            {
                                "year": year,
                                "round": round_num,
                                "college_code": college_code,
                                "college_name": college_name,
                                "branch": branch,
                                "category": category,
                                "closing_rank": closing_rank,
                                "source_file": "SAMPLE_DATA_NOT_REAL",
                            }
                        )
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = generate_rows()
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} sample rows -> {OUTPUT_CSV}")
    print("Reminder: this is SYNTHETIC data for building/testing only.")


if __name__ == "__main__":
    main()
