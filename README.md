# KCET Cutoff Predictor — Project Guide

This project is split into 4 numbered folders. Work through them **in
order** — each folder's output is the next folder's input. Every folder
has its own `README.md` explaining, in plain language:
- what that part does and why it exists
- every input file it needs
- every output file it produces
- how to run it
- what could go wrong

## The 4 parts

```
01_data_collection    →  02_data_processing   →  03_model_development  →  04_application
(get raw PDFs/data)      (clean into 1 table)     (train ML model)         (Streamlit website)
```

## Important limitation (read this first)

KEA's website (cetonline.karnataka.gov.in) cannot be reached from Claude's
sandbox — only a small allowlist of domains (github, pypi, etc.) is
reachable here. This means:

- `01_data_collection/scraper.py` is **real, working code** — but I
  cannot run it myself in this sandbox. You will run it on your own
  computer, where it will hit the real KEA site.
- So that we can build and test Parts 2, 3, and 4 today without waiting,
  `01_data_collection/generate_sample_data.py` creates a realistic
  **sample dataset** (same shape/columns real scraped data will have,
  values based on real 2021–2025 KCET cutoff ranges I looked up). You
  swap this for real scraped data later — nothing downstream changes.

## Setup (do this once)

```bash
cd kcet-cutoff-predictor
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Order of operations

1. `cd 01_data_collection` → follow its README
2. `cd 02_data_processing` → follow its README
3. `cd 03_model_development` → follow its README
4. `cd 04_application` → follow its README → `streamlit run app.py`

Go into `01_data_collection` now and open its `README.md`.
