import pandas as pd
from pathlib import Path
from datetime import datetime
from data_identification import LiteratureIdentification
from data_screening import DataScreening
from data_plots import LiteraturePlots


def main():

    # ----------------------------------------------------------
    # PATHS
    # ----------------------------------------------------------
    BASE_DIR  = Path("/home/amath/Desktop/review_v1")
    DATA_DIR  = BASE_DIR / "data"
    run_stamp = datetime.now().strftime("run_%b-%d-%Y_%Hh%M")
    OUT_DIR   = BASE_DIR / "output" / run_stamp
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Run output folder: {OUT_DIR}")

    # ----------------------------------------------------------
    # STEP 01 — DATA IDENTIFICATION
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 01: DATA IDENTIFICATION")
    print("=" * 60)

    identifier = LiteratureIdentification(
        pubmed_path = str(DATA_DIR / "csv-OncologyOR-set.csv"),
        scopus_path = str(DATA_DIR / "scopus_export_Apr 15-2026_b03f960e-1251-4326-81dd-cfa9cf7f512b.csv"),
        ieee_path   = str(DATA_DIR / "export2026.04.15-05.06.59.csv"),
        verbose     = False,
        out_dir     = OUT_DIR,
    )
    master = identifier.run()

    # ----------------------------------------------------------
    # STEP 02a — AI SCREENING: METHODOLOGY DIMENSION
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 02a: AI SCREENING — METHODOLOGY")
    print("=" * 60)

    method_labels = [
        "Multimodal Foundation Models",        # standalone FMs, LLMs, VLMs, pathology FMs
        "Knowledge-Grounded Models",           # FMs or GNNs coupled with KGs / ontologies
        "Agentic AI and Multi-Agent Systems",  # agent frameworks, MARL, tumor board simulation
    ]

    screener_methods = DataScreening(
        master_df  = master,
        cache_path = OUT_DIR / "step02a_cache_methods.csv",
        batch_size = 32,
    )
    master_methods = screener_methods.intelligent_thematic_tagging(
        candidate_labels = method_labels,
        threshold        = 0.25,
    )

    # ----------------------------------------------------------
    # STEP 02b — AI SCREENING: APPLICATION DIMENSION
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 02b: AI SCREENING — APPLICATION")
    print("=" * 60)

    application_labels = [
        "Tumor Diagnosis and Classification",
        "Drug Repositioning and Biomarker Discovery",
        "Treatment Optimization",
        "Prognosis and Survival Prediction",
        "Clinical Decision Support",
    ]

    screener_apps = DataScreening(
        master_df  = master,
        cache_path = OUT_DIR / "step02b_cache_applications.csv",
        batch_size = 32,
    )
    master_apps = screener_apps.intelligent_thematic_tagging(
        candidate_labels = application_labels,
        threshold        = 0.25,
    )

    # ----------------------------------------------------------
    # MERGE both dimensions into one master DataFrame
    # ----------------------------------------------------------
    master_tagged = master_methods.copy()
    for col in application_labels:
        master_tagged[col] = master_apps[col].values

    csv_path = OUT_DIR / "step02_master_with_ai_tags.csv"
    master_tagged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n    Full tagged dataset saved: {csv_path}")

    # ----------------------------------------------------------
    # STEP 03 — VISUALIZATIONS
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 03: VISUALIZATION")
    print("=" * 60)

    # Plot 1: Method co-occurrence heatmap (3×3)
    LiteraturePlots.thematic_heatmap(
        master   = master_tagged,
        themes   = method_labels,
        out_dir  = OUT_DIR,
        filename = "step03a_method_cooccurrence.png",
        title    = "Methodology Co-occurrence Heatmap",
    )

    # Plot 2: Application co-occurrence heatmap (5×5)
    LiteraturePlots.thematic_heatmap(
        master   = master_tagged,
        themes   = application_labels,
        out_dir  = OUT_DIR,
        filename = "step03b_application_cooccurrence.png",
        title    = "Application Co-occurrence Heatmap",
    )

    # Plot 3: Method × Application matrix (3×5) — key figure for the review
    LiteraturePlots.method_application_matrix(
        master             = master_tagged,
        method_labels      = method_labels,
        application_labels = application_labels,
        out_dir            = OUT_DIR,
    )

    # Plot 4: Temporal trends per method
    LiteraturePlots.temporal_trends(
        master   = master_tagged,
        themes   = method_labels,
        out_dir  = OUT_DIR,
        filename = "step03d_temporal_methods.png",
        title    = "Temporal Trends — Methodology",
    )

    # Plot 5: Category counts — methodology
    LiteraturePlots.category_counts(
        master   = master_tagged,
        labels   = method_labels,
        out_dir  = OUT_DIR,
        filename = "step03e_counts_methods.png",
        title    = "Papers per Methodology Category",
        color    = "#4C72B0",
    )

    # Plot 6: Category counts — applications
    LiteraturePlots.category_counts(
        master   = master_tagged,
        labels   = application_labels,
        out_dir  = OUT_DIR,
        filename = "step03f_counts_applications.png",
        title    = "Papers per Application Category",
        color    = "#16a085",
    )

    # ----------------------------------------------------------
    # DONE
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline finished successfully!")
    print(f"  All outputs saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()