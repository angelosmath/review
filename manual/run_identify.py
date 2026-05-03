import sys
import json
from pathlib import Path
from datetime import datetime

# Allow imports from review/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_identification import LiteratureIdentification

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent   # phd/
DATA_DIR  = BASE_DIR / "data"
run_stamp = datetime.now().strftime("run_%b-%d-%Y_%Hh%M")
OUT_DIR   = BASE_DIR / "output" / run_stamp
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LABELS (fill these in the exported CSV) ──────────────────
METHOD_LABELS = [
    "Multimodal Foundation Models",
    "Knowledge-Grounded Models",
    "Agentic AI and Multi-Agent Systems",
]
APPLICATION_LABELS = [
    "Tumor Diagnosis and Classification",
    "Treatment Optimization",
    "Prognosis and Survival Prediction",
    "Tumor Board Simulation and Workflow Integration",
]

# ── STEP 01: IDENTIFICATION ──────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 01: DATA IDENTIFICATION")
print("=" * 60)

identifier = LiteratureIdentification(
    pubmed_path=str(DATA_DIR / "csv-OncologyOR-set.csv"),
    scopus_path=str(DATA_DIR / "scopus_export_Apr 15-2026_b03f960e-1251-4326-81dd-cfa9cf7f512b.csv"),
    ieee_path=str(DATA_DIR / "export2026.04.15-05.06.59.csv"),
    verbose=False,
    out_dir=OUT_DIR,
)
master = identifier.run()

# ── PRISMA COUNTS (saved for run_visualize.py) ───────────────
_total    = len(master)
_dedup    = master["dedup_key"].nunique()
_dupes    = _total - _dedup
_rev_conf = int(master.loc[master["flag_review_conf"], "dedup_key"].nunique())
_screening = _dedup - _rev_conf

counts = {
    "pubmed_n":              int((master["Source"] == "PubMed").sum()),
    "scopus_n":              int((master["Source"] == "Scopus").sum()),
    "ieee_n":                int((master["Source"] == "IEEE").sum()),
    "total_raw":             _total,
    "duplicates_removed":    _dupes,
    "after_dedup":           _dedup,
    "reviews_conf_excluded": _rev_conf,
    "for_screening":         _screening,
}

counts_path = OUT_DIR / "prisma_counts.json"
with open(counts_path, "w") as f:
    json.dump(counts, f, indent=2)
print(f"\n  PRISMA counts saved: {counts_path}")

# ── EXPORT TEMPLATE FOR MANUAL SCREENING ────────────────────
screening_df = (
    master[~master["flag_review_conf"]]
    .drop_duplicates(subset=["dedup_key"])
    .copy()
)
screening_df = screening_df[["Title", "Authors", "Year", "DOI", "Source"]].copy()

for label in METHOD_LABELS + APPLICATION_LABELS:
    screening_df[label] = ""
screening_df["title_screen_pass"] = ""

template_path = OUT_DIR / "manual_screening_template.csv"
screening_df.to_csv(template_path, index=False, encoding="utf-8-sig")

print(f"\n{'=' * 60}")
print(f"  Exported {len(screening_df)} titles for manual screening:")
print(f"  {template_path}")
print(f"""
  HOW TO USE:
    1. Open manual_screening_template.csv in Excel / LibreOffice
    2. For each method/application column enter 1 (yes) or 0 (no)
    3. Set title_screen_pass to 1 (include) or 0 (exclude)
    4. Save the file (keep it as CSV)
    5. Run:  python manual/run_visualize.py
""")
print("=" * 60)
