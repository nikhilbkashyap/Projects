"""
KCET Cutoff Scraper
====================
WHAT THIS FILE DOES:
  1. Visits the official KEA website
  2. Finds links to cutoff PDF files (they publish one PDF per round,
     per year, e.g. "KCET 2025 Round 1 Engineering Cutoff")
  3. Downloads those PDFs into raw_pdfs/
  4. Opens each PDF and pulls out the cutoff table (college, branch,
     category, opening rank, closing rank)
  5. Saves everything into one combined CSV: output/scraped_cutoffs.csv

WHY IT'S SPLIT THIS WAY:
  Downloading and parsing are separated into two functions on purpose.
  KEA's PDF table layout has changed across years before, so parsing is
  the part most likely to need tweaking. If parsing breaks, you don't
  want to have to re-download everything - the PDFs stay in raw_pdfs/
  and you just fix parse_pdf() and re-run.

HOW TO RUN THIS:
  python scraper.py

  This CANNOT be run inside Claude's sandbox (the KEA website isn't on
  the reachable domain list there). Run it on your own laptop where you
  have normal internet access.

WHAT TO CHECK IF IT BREAKS:
  - KEA occasionally changes their site structure. If find_cutoff_pdf_links()
    returns an empty list, open cetonline.karnataka.gov.in in a browser,
    find the cutoff PDF links manually, and hardcode the URLs into
    MANUAL_PDF_URLS below as a fallback.
  - If parse_pdf() returns garbage/empty rows, the table structure in that
    year's PDF is different. Print pdf.pages[0].extract_table() for one
    page and look at the raw structure before adjusting the column mapping.
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
import pdfplumber
import csv

BASE_URL = "https://cetonline.karnataka.gov.in"
RAW_PDF_DIR = os.path.join(os.path.dirname(__file__), "raw_pdfs")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "scraped_cutoffs.csv")

# Fallback / seed list: real, confirmed KEA PDF URLs for the last 5 years
# (2021-2025), all rounds, found via search on 2026-08-11 (sources:
# engineering.careers360.com and giraffe-learning.com, both of which
# link directly to cetonline.karnataka.gov.in - not third-party mirrors).
#
# KEA's file-naming convention has changed year to year (a real, annoying
# fact about this site) - notice 2021/2022 use lowercase folder rounds
# (R1/R2/R3), 2023 uses a shorter ENGG_CUTOFF_2023 prefix, 2024 adds
# _GEN, and 2025-2026 switched to the PROF_CODE_E_<R|H>_<DATE> pattern.
# This inconsistency is exactly why parse_pdf() below needs per-year
# checking - a layout change often comes with a naming change too.
MANUAL_PDF_URLS = [
    # --- 2026 ---
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2026/PROF_CODE_E_R_15072026english.pdf",  # Final Round 1
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2026/PROF_CODE_E_H_15072026english.pdf",
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2026/PROF_CODE_E_R_13072026english.pdf",  # Round 1
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2026/PROF_CODE_E_H_13072026english.pdf",
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2026/PROF_CODE_E_r_06072026english.pdf",  # Mock
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2026/PROF_CODE_E_h_06072026english.pdf",
    # --- 2025 ---
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2025/PROF_CODE_e_Renglish.pdf",             # Mock
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2025/PROF_CODE_E_R_R1english.pdf",          # Round 1
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2025/PROF_CODE_E_R_30082025english.pdf",    # Round 2
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2025/PROF_CODE_E_R_11092025english.pdf",    # Final Round
    # --- 2024 ---
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2024/ENGG_CUTOFF_2024_GENenglish.pdf",      # Mock
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2024/ENGG_CUTOFF_2024_GEN_R1english.pdf",   # Round 1
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2024/ENGG_CUTOFF_2024_GEN_R2_FIN.pdf",      # Round 2
    "https://cetonline.karnataka.gov.in/keawebentry456/ugcet2024/ENGG_CUTOFF_2024_GEN_EXT_RNDenglish.pdf",  # Final/Extended
    # --- 2023 ---
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2023/ENGG_CUTOFF_2023english.pdf",            # Mock
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2023/ENGG_CUTOFF_2023_GENenglish.pdf",        # Round 1
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2023/ENGG_CUTOFF_2023_R2english.pdf",         # Round 2
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2023/ENR2_CUTGENenglish.pdf",                 # Final Round
    # --- 2022 ---
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2022/mock/engg_cutoff_gen.pdf",
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2022/R1/engg_cutoff_gen.pdf",
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2022/R2/engg_cutoff_gen.pdf",
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2022/R3/engg_cutoff_gen.pdf",
    # --- 2021 ---
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2021/mock/engg_cutoff_gen.pdf",
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2021/R1/engg_cutoff_gen.pdf",
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2021/R2/engg_cutoff_gen.pdf",
    "https://cetonline.karnataka.gov.in/keawebentry456/cet2021/R3/engg_cutoff_gen.pdf",
]


def build_predictable_url(year, dd_mm_yyyy, region="R"):
    """
    Constructs a KEA cutoff PDF URL from a known release date, using the
    naming pattern discovered above. Useful once a new round's release
    date is announced but before you've found the link on the site -
    saves you from re-scraping the links page every round.

    year: e.g. 2026
    dd_mm_yyyy: the release date as it appears in the filename, e.g. "15072026"
    region: "R" for general, "H" for Hyderabad-Karnataka (371J) quota
    """
    return (
        f"{BASE_URL}/keawebentry456/ugcet{year}/"
        f"PROF_CODE_E_{region}_{dd_mm_yyyy}english.pdf"
    )

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def find_cutoff_pdf_links():
    """
    Scans the KEA homepage / notifications page for links that look like
    cutoff PDFs. Government sites often bury these under a "Notifications"
    or "KCET" menu, so this checks a couple of likely entry points.
    """
    pdf_links = set()
    candidate_pages = [BASE_URL, f"{BASE_URL}/kea/"]

    for page_url in candidate_pages:
        try:
            resp = requests.get(page_url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  Could not reach {page_url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            link_text = a.get_text(strip=True).lower()
            if href.lower().endswith(".pdf") and (
                "cutoff" in href.lower()
                or "cut off" in link_text
                or "cutoff" in link_text
            ):
                full_url = href if href.startswith("http") else BASE_URL + href
                pdf_links.add(full_url)

    pdf_links.update(MANUAL_PDF_URLS)
    return sorted(pdf_links)


def download_pdfs(pdf_urls):
    """Downloads each PDF to raw_pdfs/ if not already downloaded."""
    os.makedirs(RAW_PDF_DIR, exist_ok=True)
    saved_paths = []

    for url in pdf_urls:
        filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", url.split("/")[-1])
        local_path = os.path.join(RAW_PDF_DIR, filename)

        if os.path.exists(local_path):
            print(f"  Already downloaded: {filename}")
            saved_paths.append(local_path)
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
            print(f"  Downloaded: {filename}")
            saved_paths.append(local_path)
            time.sleep(1)  # be polite to the server
        except requests.RequestException as e:
            print(f"  FAILED to download {url}: {e}")

    return saved_paths


def parse_pdf(pdf_path):
    """
    Extracts cutoff rows from one PDF. Expects a table with columns
    roughly like: College Code | College Name | Branch | Category | Rank

    Returns a list of dicts, one per row.
    NOTE: This is the part most likely to need adjusting per year's PDF
    layout - inspect a sample page if rows come out empty or shifted.
    """
    rows = []
    filename = os.path.basename(pdf_path)

    # Try to guess year and round from the filename - adjust the regex
    # if KEA's naming convention differs from what you see.
    year_match = re.search(r"20\d{2}", filename)
    round_match = re.search(r"[Rr]ound[_\s]?(\d)", filename)
    year = year_match.group(0) if year_match else "unknown"
    round_num = round_match.group(1) if round_match else "unknown"

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            for row in table[1:]:  # skip header row
                if not row or len(row) < 4:
                    continue
                cleaned = [c.strip() if c else "" for c in row]
                rows.append(
                    {
                        "year": year,
                        "round": round_num,
                        "college_code": cleaned[0] if len(cleaned) > 0 else "",
                        "college_name": cleaned[1] if len(cleaned) > 1 else "",
                        "branch": cleaned[2] if len(cleaned) > 2 else "",
                        "category": cleaned[3] if len(cleaned) > 3 else "",
                        "closing_rank": cleaned[4] if len(cleaned) > 4 else "",
                        "source_file": filename,
                    }
                )
    return rows


def main():
    print("Step 1/3: Finding cutoff PDF links on KEA site...")
    pdf_urls = find_cutoff_pdf_links()
    print(f"  Found {len(pdf_urls)} PDF link(s).")

    if not pdf_urls:
        print("  No links found automatically. Add URLs to MANUAL_PDF_URLS "
              "in this script and re-run.")
        return

    print("\nStep 2/3: Downloading PDFs...")
    pdf_paths = download_pdfs(pdf_urls)

    print("\nStep 3/3: Parsing PDFs into one CSV...")
    all_rows = []
    for path in pdf_paths:
        parsed = parse_pdf(path)
        print(f"  {os.path.basename(path)}: extracted {len(parsed)} rows")
        all_rows.extend(parsed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if all_rows:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nDone. Saved {len(all_rows)} rows to {OUTPUT_CSV}")
    else:
        print("\nNo rows extracted. Inspect a PDF manually - see the "
              "'WHAT TO CHECK IF IT BREAKS' notes at the top of this file.")


if __name__ == "__main__":
    main()
