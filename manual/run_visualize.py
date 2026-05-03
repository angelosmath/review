import sys
import json
from pathlib import Path

# Allow imports from review/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from data_plots import LiteraturePlots

# ── CONFIGURE HERE ───────────────────────────────────────────
# Set this to the run folder printed by run_identify.py
RUN_DIR = Path("/home/aggelos/Desktop/phd/output/run_May-03-2026_14h30")
# ────────────────────────────────────────────────────────────

COMPLETED_CSV = RUN_DIR / "manual_screening_template.csv"
PRISMA_JSON   = RUN_DIR / "prisma_counts.json"
OUT_DIR       = RUN_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

print(f"\n  Run folder: {RUN_DIR}")

# ── LOAD DATA ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LOADING COMPLETED SCREENING CSV")
print("=" * 60)

if not COMPLETED_CSV.exists():
    raise FileNotFoundError(f"Completed CSV not found:\n  {COMPLETED_CSV}\nRun run_identify.py first.")

if not PRISMA_JSON.exists():
    raise FileNotFoundError(f"PRISMA counts not found:\n  {PRISMA_JSON}\nRun run_identify.py first.")

df = pd.read_csv(COMPLETED_CSV)

with open(PRISMA_JSON) as f:
    counts = json.load(f)

# Coerce all label columns to integer (handles blank cells gracefully)
all_labels = METHOD_LABELS + APPLICATION_LABELS
for col in all_labels + ["title_screen_pass"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

screened        = df[df["title_screen_pass"] == 1].copy()
_title_passed   = len(screened)
_title_excluded = len(df) - _title_passed

print(f"\n  Total papers loaded : {len(df)}")
print(f"  Title screen passed : {_title_passed}")
print(f"  Title screen excluded: {_title_excluded}")

# ── STEP 03: VISUALIZATIONS ──────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 03: VISUALIZATIONS")
print("=" * 60)

# PRISMA flowchart
LiteraturePlots.prisma_flowchart(
    **counts,
    title_screen_passed=_title_passed,
    title_screen_excluded=_title_excluded,
    out_dir=OUT_DIR,
    filename="prisma_flowchart.png",
)

# Category counts — all papers (with method/app tags)
LiteraturePlots.category_counts(
    master=df, labels=METHOD_LABELS,
    out_dir=OUT_DIR, filename="counts_methods.png",
    title="Papers per Methodology Category", color="#4C72B0",
)
LiteraturePlots.category_counts(
    master=df, labels=APPLICATION_LABELS,
    out_dir=OUT_DIR, filename="counts_applications.png",
    title="Papers per Application Category", color="#16a085",
)

# Plots that use screened papers only
if _title_passed > 0:
    LiteraturePlots.temporal_trends(
        master=screened, themes=METHOD_LABELS,
        out_dir=OUT_DIR, filename="temporal_methods.png",
        title="Temporal Trends — Methodology",
    )

    LiteraturePlots.venn_rq_overlap(
        master=screened, method_labels=METHOD_LABELS,
        out_dir=OUT_DIR,
    )

    LiteraturePlots.distribution_pie(
        master=screened, labels=METHOD_LABELS,
        out_dir=OUT_DIR, filename="pie_methods.png",
        title=f"Methodology Distribution (n = {_title_passed})",
        colors=["#4C72B0", "#DD8452", "#55A868"],
    )

    LiteraturePlots.distribution_pie(
        master=screened, labels=APPLICATION_LABELS,
        out_dir=OUT_DIR, filename="pie_applications.png",
        title=f"Application Distribution (n = {_title_passed})",
        colors=["#C44E52", "#8172B2", "#937860", "#DA8BC3"],
    )

    LiteraturePlots.method_application_matrix(
        master=screened,
        method_labels=METHOD_LABELS,
        application_labels=APPLICATION_LABELS,
        out_dir=OUT_DIR,
    )
else:
    print("\n  No screened papers — skipping screened-only plots.")

# ── STEP 04: EXPORT ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 04: EXPORT FOR ABSTRACT SCREENING")
print("=" * 60)

keep_cols = ["Title", "Authors", "Year", "DOI", "Source"] + all_labels
export_path = OUT_DIR / "papers_for_abstract_screening.csv"
screened[keep_cols].to_csv(export_path, index=False, encoding="utf-8-sig")
print(f"\n  Papers for abstract screening : {_title_passed}")
print(f"  CSV saved                     : {export_path}")

print("\n" + "=" * 60)
print("  Done! All outputs saved to:")
print(f"  {OUT_DIR}")
print("=" * 60)
