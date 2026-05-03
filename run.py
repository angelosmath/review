import pandas as pd
from pathlib import Path
from datetime import datetime
from data_identification import LiteratureIdentification
from data_plots import LiteraturePlots
# from data_screening import DataScreening  # re-enable when running AI screening (Step 02a/02b)


def main():

    # ----------------------------------------------------------
    # PATHS
    # ----------------------------------------------------------
    BASE_DIR   = Path(__file__).resolve().parent.parent
    DATA_DIR   = BASE_DIR / "data"
    OUT_DIR    = BASE_DIR / "output"          # fixed — Phase 1 checkpoint lives here
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    STEP01_CSV = OUT_DIR / "step01_master_flagged.csv"

    _manual_matches = sorted(DATA_DIR.glob("annotated_manual_screening_completed*.csv"))
    MANUAL_CSV = _manual_matches[0] if _manual_matches else DATA_DIR / "annotated_manual_screening_completed.csv"

    # ----------------------------------------------------------
    # LABELS — defined once, shared by Phase 1 (template) and Phase 2 (plots)
    # ----------------------------------------------------------
    method_labels = [
        "Multimodal Foundation Models",        # standalone FMs, LLMs, VLMs, pathology FMs
        "Knowledge-Grounded Models",           # FMs coupled with GNNs, KGs / ontologies
        "Agentic AI and Multi-Agent Systems",  # agent frameworks, MARL, tumor board simulation
    ]
    application_labels = [
        "Tumor Diagnosis and Classification",          # pathology, imaging, biomarker-based dx
        "Treatment Optimization",                      # drug selection, dosing, radiotherapy planning
        "Prognosis and Survival Prediction",           # outcome prediction, risk stratification
        "Tumor Board Simulation and Workflow Integration",  # MDT automation, clinical decision support
    ]

    # ----------------------------------------------------------
    # PHASE 1 — DATA IDENTIFICATION
    # Runs once, saves checkpoint + annotation template, then stops.
    # ----------------------------------------------------------
    if not STEP01_CSV.exists():
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

        _save_annotation_template(master, method_labels, application_labels, MANUAL_CSV)

        print("\n" + "=" * 60)
        print("  PHASE 1 COMPLETE")
        print(f"  Identification checkpoint : {STEP01_CSV}")
        print(f"  Annotation template saved : {MANUAL_CSV}")
        print()
        print("  ► Open the template CSV, fill in 1 or 0 for each column, save, and re-run.")
        print("=" * 60)
        return

    # ----------------------------------------------------------
    # Phase 1 checkpoint — load from disk, skip identification
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 01: Identification checkpoint found — loading from disk")
    print("=" * 60)

    master = pd.read_csv(STEP01_CSV)
    print(f"    Records loaded : {len(master)}")

    _total     = len(master)
    _dedup     = master["dedup_key"].nunique()
    _dupes     = _total - _dedup
    _rev_conf  = int(master.loc[master["flag_review_conf"], "dedup_key"].nunique())
    _screening = _dedup - _rev_conf

    # ----------------------------------------------------------
    # PHASE 2 gate — generate template if missing, then stop
    # ----------------------------------------------------------
    if not MANUAL_CSV.exists():
        _save_annotation_template(master, method_labels, application_labels, MANUAL_CSV)

        print("\n" + "=" * 60)
        print("  ANNOTATION TEMPLATE GENERATED")
        print(f"  Saved to : {MANUAL_CSV}")
        print()
        print("  ► Open the template CSV, fill in 1 or 0 for each column, save, and re-run.")
        print("=" * 60)
        return

    # ----------------------------------------------------------
    # PHASE 2 — timestamped run folder for all plots and exports
    # ----------------------------------------------------------
    run_stamp = datetime.now().strftime("run_%b-%d-%Y_%Hh%M")
    RUN_DIR   = OUT_DIR / run_stamp
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Phase 2 output folder: {RUN_DIR}")

    # ----------------------------------------------------------
    # STEP 02 — LOAD MANUAL ANNOTATIONS
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 02: LOAD MANUAL ANNOTATIONS")
    print("=" * 60)

    master_tagged = pd.read_csv(MANUAL_CSV)
    print(f"\n    Manual annotations loaded : {MANUAL_CSV}")
    print(f"    Total records             : {len(master_tagged)}")

    screened        = master_tagged[
        master_tagged[method_labels].eq(1).any(axis=1) &
        master_tagged["Not Relevant"].ne(1)
    ].copy()
    _title_passed   = len(screened)
    _title_excluded = len(master_tagged) - _title_passed

    print(f"    Papers included (≥1 method label = 1) : {_title_passed}")
    print(f"    Papers excluded                       : {_title_excluded}")

    """
    # ----------------------------------------------------------
    # STEP 02a — AI SCREENING: METHODOLOGY DIMENSION  [BYPASSED]
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 02a: AI SCREENING — METHODOLOGY")
    print("=" * 60)

    _model      = DataScreening.FAST_MODEL if TEST_MODE else None   # None → DEFAULT_MODEL
    _batch_size = 8 if TEST_MODE else 32
    _threshold  = 0.25

    screener_methods = DataScreening(
        master_df  = master,
        cache_path = RUN_DIR / "step02a_cache_methods.csv",
        model_name = _model,
        batch_size = _batch_size,
    )
    master_methods = screener_methods.intelligent_thematic_tagging(
        candidate_labels = method_labels,
        threshold        = _threshold,
    )

    # ----------------------------------------------------------
    # STEP 02b — AI SCREENING: APPLICATION DIMENSION  [BYPASSED]
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 02b: AI SCREENING — APPLICATION")
    print("=" * 60)

    screener_apps = DataScreening(
        master_df  = master,
        cache_path = RUN_DIR / "step02b_cache_applications.csv",
        model_name = _model,
        batch_size = _batch_size,
    )
    master_apps = screener_apps.intelligent_thematic_tagging(
        candidate_labels = application_labels,
        threshold        = _threshold,
    )

    # ----------------------------------------------------------
    # MERGE both dimensions into one master DataFrame  [BYPASSED]
    # ----------------------------------------------------------
    app_cols = ["dedup_key"] + application_labels + [f"{l}_score" for l in application_labels]
    master_tagged = master_methods.merge(
        master_apps[app_cols],
        on="dedup_key",
        how="left",
        suffixes=("", "_app"),
    )

    csv_path = RUN_DIR / "step02_master_with_ai_tags.csv"
    master_tagged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\\n    Full tagged dataset saved: {csv_path}")

    # ----------------------------------------------------------
    # STEP 02c — TITLE SCREENING (RQ relevance filter)  [BYPASSED]
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 02c: TITLE SCREENING")
    print("=" * 60)

    SCREEN_THRESHOLD = 0.75
    score_cols = [f"{l}_score" for l in method_labels]
    master_tagged["title_screen_pass"] = (
        master_tagged[score_cols]
        .gt(SCREEN_THRESHOLD)
        .any(axis=1)
    )

    screened        = master_tagged[master_tagged["title_screen_pass"]].copy()
    _title_passed   = len(screened)
    _title_excluded = len(master_tagged) - _title_passed

    print(f"\\n    Screening threshold   : {SCREEN_THRESHOLD}")
    print(f"    Papers before screen  : {len(master_tagged)}")
    print(f"    Papers passed         : {_title_passed}")
    print(f"    Papers excluded       : {_title_excluded}")
    """

    # ----------------------------------------------------------
    # STEP 03 — VISUALIZATIONS  (all saved to RUN_DIR)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 03: VISUALIZATION")
    print("=" * 60)

    # Plot 1: Method co-occurrence heatmap (3×3)
    # LiteraturePlots.thematic_heatmap(
    #     master   = master_tagged,
    #     themes   = method_labels,
    #     out_dir  = RUN_DIR,
    #     filename = "step03a_method_cooccurrence.png",
    #     title    = "Methodology Co-occurrence Heatmap",
    # )

    # Plot 2: Application co-occurrence heatmap (4×4)
    # LiteraturePlots.thematic_heatmap(
    #     master   = master_tagged,
    #     themes   = application_labels,
    #     out_dir  = RUN_DIR,
    #     filename = "step03b_application_cooccurrence.png",
    #     title    = "Application Co-occurrence Heatmap",
    # )

    # Plot 3: Method × Application matrix (3×4) — key figure for the review
    # LiteraturePlots.method_application_matrix(
    #     master             = master_tagged,
    #     method_labels      = method_labels,
    #     application_labels = application_labels,
    #     out_dir            = RUN_DIR,
    # )

    # Plot 4: Temporal trends per method
    LiteraturePlots.temporal_trends(
        master   = master_tagged,
        themes   = method_labels,
        out_dir  = RUN_DIR,
        filename = "step03d_temporal_methods.png",
        title    = "Temporal Trends — Methodology",
    )

    # Plot 5: Category counts — methodology
    LiteraturePlots.category_counts(
        master   = master_tagged,
        labels   = method_labels,
        out_dir  = RUN_DIR,
        filename = "step03e_counts_methods.png",
        title    = "Papers per Methodology Category",
        color    = "#4C72B0",
    )

    # Plot 6: Category counts — applications
    LiteraturePlots.category_counts(
        master   = master_tagged,
        labels   = application_labels,
        out_dir  = RUN_DIR,
        filename = "step03f_counts_applications.png",
        title    = "Papers per Application Category",
        color    = "#16a085",
    )

    # Plot 7: Venn diagram — RQ overlap among screened papers
    LiteraturePlots.venn_rq_overlap(
        master        = screened,
        method_labels = method_labels,
        out_dir       = RUN_DIR,
    )

    # Plot 8: PRISMA flow diagram
    LiteraturePlots.prisma_flowchart(
        pubmed_n               = int((master["Source"] == "PubMed").sum()),
        scopus_n               = int((master["Source"] == "Scopus").sum()),
        ieee_n                 = int((master["Source"] == "IEEE").sum()),
        total_raw              = _total,
        duplicates_removed     = _dupes,
        after_dedup            = _dedup,
        reviews_conf_excluded  = _rev_conf,
        for_screening          = _screening,
        title_screen_passed    = _title_passed,
        title_screen_excluded  = _title_excluded,
        out_dir                = RUN_DIR,
        filename               = "step00_prisma_flowchart_full.png",
    )

    # Plot 9: Donut chart — methodology distribution (screened papers only)
    LiteraturePlots.distribution_pie(
        master   = screened,
        labels   = method_labels,
        out_dir  = RUN_DIR,
        filename = "step03g_pie_methods.png",
        title    = "Methodology Distribution (n = {})".format(_title_passed),
        colors   = ["#4C72B0", "#DD8452", "#55A868"],
    )

    # Plot 10: Donut chart — application distribution (screened papers only)
    LiteraturePlots.distribution_pie(
        master   = screened,
        labels   = application_labels,
        out_dir  = RUN_DIR,
        filename = "step03h_pie_applications.png",
        title    = "Application Distribution (n = {})".format(_title_passed),
        colors   = ["#C44E52", "#8172B2", "#937860", "#DA8BC3"],
    )

    # Plot 11: Cancer type distribution
    LiteraturePlots.cancer_type_distribution(master=screened, out_dir=RUN_DIR)

    # ----------------------------------------------------------
    # STEP 04 — EXPORT  (saved to RUN_DIR)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 04: EXPORT FOR ABSTRACT SCREENING")
    print("=" * 60)

    keep_cols = ["Title", "Authors", "Year", "DOI", "Source"] \
                + method_labels + application_labels

    screening_csv = RUN_DIR / "step04_papers_for_abstract_screening.csv"
    screened[keep_cols].to_csv(screening_csv, index=False, encoding="utf-8-sig")
    print(f"\n    Papers for abstract screening : {_title_passed}")
    print(f"    CSV saved                     : {screening_csv}")

    """
    # ----------------------------------------------------------
    # STEP 05 — VALIDATION: threshold & recall checks  [BYPASSED]
    # Requires *_score columns produced by the DeBERTa model.
    # Re-enable when running the AI screening pipeline.
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 05: VALIDATION EXPORT")
    print("=" * 60)

    val_cols = ["Title", "Authors", "Year", "DOI", "Source"] + score_cols

    excluded = master_tagged[~master_tagged["title_screen_pass"]].copy()
    excluded["_max_score"] = excluded[score_cols].max(axis=1)

    borderline = excluded[excluded["_max_score"] >= 0.80].copy()
    borderline = borderline.sort_values("_max_score", ascending=False)
    borderline["Your_Decision"] = ""
    borderline["Notes"]         = ""

    borderline_csv = RUN_DIR / "step05a_borderline_papers.csv"
    borderline[val_cols + ["_max_score", "Your_Decision", "Notes"]].to_csv(
        borderline_csv, index=False, encoding="utf-8-sig"
    )
    print(f"\\n    5a — Borderline papers (score 0.80–0.92) : {len(borderline)}")
    print(f"         CSV : {borderline_csv}")

    deep_excluded = excluded[excluded["_max_score"] < 0.80].copy()
    sample_n = min(30, len(deep_excluded))
    random_sample = deep_excluded.sample(sample_n, random_state=42)
    random_sample = random_sample.sort_values("_max_score", ascending=False)
    random_sample["Your_Decision"] = ""
    random_sample["Notes"]         = ""

    random_csv = RUN_DIR / "step05b_random_excluded_sample.csv"
    random_sample[val_cols + ["_max_score", "Your_Decision", "Notes"]].to_csv(
        random_csv, index=False, encoding="utf-8-sig"
    )
    print(f"\\n    5b — Random excluded sample : {sample_n} / {len(deep_excluded)}")
    print(f"         CSV : {random_csv}")
    """

    # ----------------------------------------------------------
    # DONE
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline finished successfully!")
    print(f"  All outputs saved to: {RUN_DIR}")
    print("=" * 60)


# ----------------------------------------------------------
# HELPER — builds and saves the annotation template CSV
# ----------------------------------------------------------
def _save_annotation_template(master, method_labels, application_labels, path):
    template = (
        master[~master["flag_review_conf"]]
        .drop_duplicates(subset=["dedup_key"], keep="first")
        .copy()
    )
    for col in method_labels + application_labels:
        template[col] = 0
    template["Not Relevant"] = 0   # 1 = exclude this paper from all analysis
    template["Cancer Type"]  = ""

    cols = ["Title", "Authors", "Year", "DOI", "Source"] \
           + method_labels + application_labels + ["Not Relevant", "Cancer Type"]
    template[cols].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"    Annotation template : {path}  ({len(template)} papers)")


if __name__ == "__main__":
    main()
