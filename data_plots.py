import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path


class LiteraturePlots:

    # ----------------------------------------------------------
    # PLOT 1: Identification summary bar chart
    # ----------------------------------------------------------
    @staticmethod
    def identification_summary(master: pd.DataFrame, out_dir: Path | None = None):

        if "dedup_key" not in master.columns:
            raise ValueError("dedup_key is missing. Run _deduplicate() first.")

        total    = len(master)
        pubmed   = int((master["Source"] == "PubMed").sum())
        scopus   = int((master["Source"] == "Scopus").sum())
        ieee     = int((master["Source"] == "IEEE").sum())

        dedup_papers       = int(master["dedup_key"].nunique())
        duplicates_removed = int(total - dedup_papers)

        excluded_papers = (
            int(master.loc[master["flag_review_conf"], "dedup_key"].nunique())
            if "flag_review_conf" in master.columns else 0
        )
        kept_papers = int(dedup_papers - excluded_papers)

        labels = [
            "PubMed", "Scopus", "IEEE",
            "Total\n(raw)",
            "Dupes\nremoved",
            "Dedup\npapers",
            "Excluded\n(rev+conf)",
            "Kept\n(final)"
        ]
        values = [
            pubmed, scopus, ieee,
            total,
            duplicates_removed,
            dedup_papers,
            excluded_papers,
            kept_papers
        ]
        colors = [
            "#4C72B0", "#A1ADC2", "#373B42",
            "#2ecc71", "#e74c3c", "#27ae60",
            "#f39c12", "#16a085"
        ]

        fig, ax = plt.subplots(figsize=(13, 6))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(val),
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

        legend_patches = [
            mpatches.Patch(color="#4C72B0", label="PubMed records"),
            mpatches.Patch(color="#A1ADC2", label="Scopus records"),
            mpatches.Patch(color="#373B42", label="IEEE records"),
            mpatches.Patch(color="#2ecc71", label="Total records (raw)"),
            mpatches.Patch(color="#e74c3c", label="Duplicates removed"),
            mpatches.Patch(color="#27ae60", label="Unique papers (dedup)"),
            mpatches.Patch(color="#f39c12", label="Excluded (review+conf)"),
            mpatches.Patch(color="#16a085", label="Kept papers (final)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
        ax.set_title("Identification Summary", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        out_dir = Path(out_dir) if out_dir is not None else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_dir / "step01_identification_summary.png", dpi=300)
        plt.close()
        print(f"    Plot saved: step01_identification_summary.png")

    # ----------------------------------------------------------
    # PLOT 2: Generic thematic co-occurrence heatmap
    # ----------------------------------------------------------
    @staticmethod
    def thematic_heatmap(
        master:   pd.DataFrame,
        themes:   list[str],
        out_dir:  Path | None = None,
        filename: str = "thematic_heatmap.png",
        title:    str = "Thematic Co-occurrence Heatmap",
    ):
        """
        Square co-occurrence heatmap for any list of binary theme columns.

        Args:
            master   : DataFrame containing one binary column per theme.
            themes   : List of theme column names (must exist in master).
            out_dir  : Directory to save the PNG.
            filename : Output filename.
            title    : Plot title.
        """
        df = master.copy()

        missing = [t for t in themes if t not in df.columns]
        if missing:
            raise ValueError(f"Theme columns missing from DataFrame: {missing}")

        # Vectorized co-occurrence via matrix multiply
        binary       = df[themes].fillna(0).astype(int).values   # (n_papers, n_themes)
        cooccurrence = binary.T @ binary                           # (n_themes, n_themes)
        matrix_int   = pd.DataFrame(cooccurrence, index=themes, columns=themes)

        fig_size = max(8, len(themes) * 1.8)
        plt.figure(figsize=(fig_size, fig_size * 0.85))
        sns.heatmap(
            matrix_int, annot=True, fmt="d", cmap="Reds", square=True,
            cbar_kws={"label": "Paper Count"},
            annot_kws={"size": 13, "weight": "bold"},
        )
        plt.title(title, fontsize=15, fontweight="bold", pad=20)
        plt.xticks(rotation=45, ha="right", fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()

        if out_dir:
            out_path = Path(out_dir) / filename
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"    Plot saved: {filename}")

    # ----------------------------------------------------------
    # PLOT 3: Method × Application matrix  ← KEY FIGURE
    # ----------------------------------------------------------
    @staticmethod
    def method_application_matrix(
        master:             pd.DataFrame,
        method_labels:      list[str],
        application_labels: list[str],
        out_dir:            Path | None = None,
    ):
        """
        Rectangular heatmap: rows = methods (3), columns = applications (5).
        Each cell = number of papers tagged with that method AND that application.
        This is the core figure answering all three RQs simultaneously.
        """
        df = master.copy()

        missing = [c for c in method_labels + application_labels if c not in df.columns]
        if missing:
            raise ValueError(f"Columns missing from DataFrame: {missing}")

        # Build the (n_methods × n_applications) matrix
        matrix = np.zeros((len(method_labels), len(application_labels)), dtype=int)
        for i, m in enumerate(method_labels):
            for j, a in enumerate(application_labels):
                matrix[i, j] = int(((df[m] == 1) & (df[a] == 1)).sum())

        matrix_df = pd.DataFrame(matrix, index=method_labels, columns=application_labels)

        # Short display names so the axis labels are readable
        method_short = {
            "Multimodal Foundation Models":        "Foundation\nModels",
            "Knowledge-Grounded Models":           "Knowledge-\nGrounded",
            "Agentic AI and Multi-Agent Systems":  "Agentic AI /\nMulti-Agent",
        }
        app_short = {
            "Tumor Diagnosis and Classification":          "Diagnosis &\nClassification",
            "Drug Repositioning and Biomarker Discovery":  "Drug Repos. &\nBiomarkers",
            "Treatment Optimization":                      "Treatment\nOptimization",
            "Prognosis and Survival Prediction":           "Prognosis &\nSurvival",
            "Tumor Board Simulation and Workflow Integration": "Tumor Board &\nWorkflow",
        }
        matrix_df.index   = [method_short.get(m, m) for m in method_labels]
        matrix_df.columns = [app_short.get(a, a) for a in application_labels]

        fig, ax = plt.subplots(figsize=(13, 5))
        sns.heatmap(
            matrix_df,
            annot=True, fmt="d", cmap="YlOrRd", square=False,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Paper Count", "shrink": 0.8},
            annot_kws={"size": 14, "weight": "bold"},
            ax=ax,
        )
        ax.set_title(
            "AI Methodology × Clinical Application",
            fontsize=15, fontweight="bold", pad=20
        )
        ax.set_xlabel("Clinical Application Domain", fontsize=12, labelpad=12)
        ax.set_ylabel("AI Methodology", fontsize=12, labelpad=12)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10, rotation=0)

        plt.tight_layout()

        if out_dir:
            out_path = Path(out_dir) / "step03c_method_x_application.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"    Plot saved: step03c_method_x_application.png")

    # ----------------------------------------------------------
    # PLOT 4: Temporal trends line chart
    # ----------------------------------------------------------
    @staticmethod
    def temporal_trends(
        master:   pd.DataFrame,
        themes:   list[str],
        out_dir:  Path | None = None,
        filename: str = "temporal_trends.png",
        title:    str = "Temporal Trends by Theme",
    ):
        """
        Line chart: number of papers per theme per year.

        Args:
            master   : DataFrame with binary theme columns and a 'Year' column.
            themes   : List of theme column names.
            out_dir  : Directory to save the PNG.
            filename : Output filename.
            title    : Plot title.
        """
        df = master.copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
        df = df[df["Year"] >= 2021]

        yearly = df.groupby("Year")[themes].sum().sort_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        markers = ["o", "s", "D", "^", "v"]
        for idx, theme in enumerate(themes):
            ax.plot(
                yearly.index, yearly[theme],
                marker=markers[idx % len(markers)],
                linewidth=2.5, markersize=7,
                label=theme,
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Number of Papers", fontsize=11)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        if out_dir:
            out_path = Path(out_dir) / filename
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"    Plot saved: {filename}")

    # ----------------------------------------------------------
    # PLOT 5: Category counts bar chart
    # ----------------------------------------------------------
    @staticmethod
    def category_counts(
        master:   pd.DataFrame,
        labels:   list[str],
        out_dir:  Path | None = None,
        filename: str = "category_counts.png",
        title:    str = "Papers per Category",
        color:    str = "#4C72B0",
    ):
        """
        Horizontal stacked bar chart — each bar split by database contribution.

        Args:
            master   : DataFrame with one binary column per label + a 'Source' column.
            labels   : List of label column names.
            out_dir  : Directory to save the PNG.
            filename : Output filename.
            title    : Plot title.
            color    : Fallback colour (used only if 'Source' column is absent).
        """
        missing = [l for l in labels if l not in master.columns]
        if missing:
            raise ValueError(f"Label columns missing from DataFrame: {missing}")

        # Sort categories by total count ascending (so largest is on top)
        total_counts = master[labels].sum().astype(int).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(11, max(4, len(labels) * 1.4)))

        if "Source" in master.columns:
            sources      = ["PubMed", "Scopus", "IEEE"]
            source_colors = {"PubMed": "#4C72B0", "Scopus": "#A1ADC2", "IEEE": "#373B42"}

            # Per-source counts for each label (keep same row order as total_counts)
            per_source = {
                src: master[master["Source"] == src][total_counts.index].sum().astype(int)
                for src in sources
            }

            lefts = pd.Series(0, index=total_counts.index)
            for src in sources:
                vals = per_source[src]
                bars = ax.barh(
                    total_counts.index, vals,
                    left=lefts,
                    color=source_colors[src],
                    edgecolor="white", linewidth=0.8,
                    height=0.6, label=src,
                )
                lefts = lefts + vals

            # Total count label at the end of each bar
            for lbl, tot in total_counts.items():
                ax.text(
                    tot + max(total_counts.values) * 0.01,
                    total_counts.index.get_loc(lbl),
                    str(tot),
                    va="center", fontsize=11, fontweight="bold",
                )

            ax.legend(loc="lower right", fontsize=10, framealpha=0.8)

        else:
            # Fallback: single-colour bars
            bars = ax.barh(total_counts.index, total_counts.values,
                           color=color, edgecolor="white", height=0.6)
            for bar, val in zip(bars, total_counts.values):
                ax.text(
                    val + max(total_counts.values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=11, fontweight="bold",
                )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Number of Papers", fontsize=11)
        ax.set_xlim(0, max(total_counts.values) * 1.18 if max(total_counts.values) > 0 else 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        if out_dir:
            out_path = Path(out_dir) / filename
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"    Plot saved: {filename}")

    # ----------------------------------------------------------
    # PLOT 6: Pie chart for category distribution
    # ----------------------------------------------------------
    @staticmethod
    def distribution_pie(
        master:   pd.DataFrame,
        labels:   list[str],
        out_dir:  Path | None = None,
        filename: str = "distribution_pie.png",
        title:    str = "Distribution",
        colors:   list[str] | None = None,
    ):
        """
        Full pie chart showing the share of papers per category.

        Args:
            master   : DataFrame with one binary column per label.
            labels   : List of label column names.
            out_dir  : Directory to save the PNG.
            filename : Output filename.
            title    : Plot title.
            colors   : Optional list of hex colours (one per label).
        """
        missing = [l for l in labels if l not in master.columns]
        if missing:
            raise ValueError(f"Label columns missing from DataFrame: {missing}")

        counts = master[labels].sum().astype(int)
        counts = counts[counts > 0].sort_values(ascending=False)

        unique_papers = int(master[labels].any(axis=1).sum())

        default_colors = ["#E63946", "#8B1A1A", "#4CAF50", "#00BCD4", "#FF9800",
                          "#9C27B0", "#3F51B5", "#795548", "#607D8B", "#F06292"]
        bar_colors = colors if colors else default_colors[: len(counts)]

        fig, ax = plt.subplots(figsize=(9, 6))

        wedges, texts = ax.pie(
            counts.values,
            labels=None,
            colors=bar_colors,
            startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
        )

        # Count labels — all outside with connecting lines
        for wedge, count in zip(wedges, counts.values):
            angle = (wedge.theta1 + wedge.theta2) / 2
            angle_rad = np.radians(angle)
            x_inner = 0.85 * np.cos(angle_rad)
            y_inner = 0.85 * np.sin(angle_rad)
            x_outer = 1.25 * np.cos(angle_rad)
            y_outer = 1.25 * np.sin(angle_rad)
            ax.annotate(
                str(count),
                xy=(x_inner, y_inner),
                xytext=(x_outer, y_outer),
                ha="center", va="center",
                fontsize=12, fontweight="bold", color="black",
                arrowprops=dict(arrowstyle="-", color="#888888", lw=1.2),
            )

        ax.legend(
            wedges, counts.index,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            fontsize=10,
            frameon=False,
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=16)

        plt.tight_layout()

        if out_dir:
            out_path = Path(out_dir) / filename
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"    Plot saved: {filename}")

    # ----------------------------------------------------------
    # PLOT 7: PRISMA 2020 flow diagram
    # ----------------------------------------------------------
    @staticmethod
    def prisma_flowchart(
        pubmed_n:              int,
        scopus_n:              int,
        ieee_n:                int,
        total_raw:             int,
        duplicates_removed:    int,
        after_dedup:           int,
        reviews_conf_excluded: int,
        for_screening:         int,
        title_screen_passed:   int | None = None,
        title_screen_excluded: int | None = None,
        out_dir:               Path | None = None,
        filename:              str = "step00_prisma_flowchart.png",
    ):
        """
        PRISMA 2020-style flow diagram matching the standard 4-stage layout:
        IDENTIFICATION → SCREENING → ELIGIBILITY → INCLUDED.
        ELIGIBILITY and INCLUDED show as pending until manual screening is complete.
        """
        include_title_screen = (title_screen_passed is not None
                                and title_screen_excluded is not None)

        fig, ax = plt.subplots(figsize=(14, 22))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 22)
        ax.axis("off")

        # ── helpers ──────────────────────────────────────────────
        MX, MW = 5.5, 5.5     # main box centre-x and width  → right edge at 8.25
        EX, EW = 12.0, 3.5    # excl box centre-x and width  → left edge at 10.25

        def mbox(y, h, text, fc="#D6EAF8", ec="#2E86C1", fs=9):
            ax.add_patch(mpatches.FancyBboxPatch(
                (MX - MW / 2, y - h / 2), MW, h,
                boxstyle="round,pad=0.15",
                facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2,
            ))
            ax.text(MX, y, text, ha="center", va="center",
                    fontsize=fs, multialignment="center", zorder=3)

        def ebox(y, h, text, fc="#FADBD8", ec="#C0392B", fs=8.5):
            ax.add_patch(mpatches.FancyBboxPatch(
                (EX - EW / 2, y - h / 2), EW, h,
                boxstyle="round,pad=0.15",
                facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2,
            ))
            ax.text(EX - EW / 2 + 0.2, y, text, ha="left", va="center",
                    fontsize=fs, multialignment="left", zorder=3)

        def stage_strip(label, y_bot, y_top, fc="#D6EAF8", ec="#2E86C1"):
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.15, y_bot + 0.15), 1.0, y_top - y_bot - 0.3,
                boxstyle="round,pad=0.1",
                facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2,
            ))
            ax.text(0.65, (y_bot + y_top) / 2, label,
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color=ec, rotation=90, zorder=3)

        def arr_down(y1, y2):
            """Arrow going downward from y1 to y2 along the main column."""
            ax.annotate("", xy=(MX, y2 + 0.08), xytext=(MX, y1 - 0.08),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5), zorder=4)

        def arr_excl(y):
            """Horizontal arrow from right edge of main box to left edge of excl box."""
            ax.annotate("", xy=(EX - EW / 2 + 0.08, y), xytext=(MX + MW / 2 - 0.08, y),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5), zorder=4)

        # ── stage strips ─────────────────────────────────────────
        stage_strip("IDENTIFICATION", 15.8, 21.8, fc="#D6EAF8", ec="#2E86C1")
        stage_strip("SCREENING",       9.8, 15.7, fc="#D5F5E3", ec="#27AE60")
        stage_strip("ELIGIBILITY",     4.5,  9.7, fc="#FEF9E7", ec="#E67E22")
        stage_strip("INCLUDED",        0.3,  4.4, fc="#FDEDEC", ec="#C0392B")

        # ── IDENTIFICATION ────────────────────────────────────────
        id_text = (f"Records identified\n"
                   f"PubMed (n = {pubmed_n})\n"
                   f"Scopus (n = {scopus_n})\n"
                   f"IEEE Xplore (n = {ieee_n})\n"
                   f"Total (n = {total_raw})")
        mbox(20.2, 2.6, id_text, fs=9)

        excl1 = (f"• Duplicates removed\n  (n = {duplicates_removed})\n"
                 f"• Reviews & conf. excluded\n  (n = {reviews_conf_excluded})")
        ebox(20.2, 1.8, excl1, fs=8.5)
        arr_excl(20.2)

        arr_down(18.9, 17.65)

        mbox(17.2, 0.9, f"Records for title screening\nn = {for_screening}")

        arr_down(16.75, 16.0)

        # ── SCREENING ─────────────────────────────────────────────
        if include_title_screen:
            mbox(15.4, 0.9,
                 f"AI title screening (NLI model)\nn = {for_screening}",
                 fc="#EBF5EB", ec="#27AE60")

            excl2 = (f"• Excluded — not relevant to RQs\n"
                     f"  (n = {title_screen_excluded})\n"
                     f"• DeBERTa-v3-large, thr. = 0.92")
            ebox(15.4, 1.3, excl2, fc="#EAFAF1", ec="#27AE60", fs=8.5)
            arr_excl(15.4)

            arr_down(14.95, 13.75)

            mbox(13.3, 0.9,
                 f"Records for abstract screening\nn = {title_screen_passed}",
                 fc="#D5F5E3", ec="#1E8449")

            arr_down(12.85, 10.15)
        else:
            mbox(13.3, 0.9,
                 f"Records for abstract screening\nn = {for_screening}",
                 fc="#D5F5E3", ec="#1E8449")
            arr_down(12.85, 10.15)

        # ── ELIGIBILITY ───────────────────────────────────────────
        mbox(9.7, 0.9,
             "Records assessed for eligibility\nn = pending abstract screening",
             fc="#FEF9E7", ec="#E67E22", fs=9)

        ebox(9.7, 1.1,
             "• Excluded after abstract screening\n  n = pending",
             fc="#FEF9E7", ec="#E67E22", fs=8.5)
        arr_excl(9.7)

        arr_down(9.25, 8.05)

        mbox(7.6, 0.9,
             "Records for full-text screening\nn = pending",
             fc="#FEF9E7", ec="#E67E22", fs=9)

        arr_down(7.15, 5.0)

        # ── INCLUDED ─────────────────────────────────────────────
        mbox(2.5, 1.2,
             "Studies included\nn = pending full-text screening",
             fc="#FDEDEC", ec="#C0392B", fs=9)

        ax.set_title("PRISMA 2020 Flow Diagram",
                     fontsize=13, fontweight="bold", pad=10)
        plt.tight_layout()

        if out_dir:
            out_path = Path(out_dir) / filename
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"    Plot saved: {filename}")

    # ----------------------------------------------------------
    # PLOT 8: RQ overlap Venn diagram
    # ----------------------------------------------------------
    @staticmethod
    def venn_rq_overlap(
        master:        pd.DataFrame,
        method_labels: list[str],
        out_dir:       Path | None = None,
        filename:      str = "step02c_venn_rq_overlap.png",
    ):
        """
        3-circle Venn diagram showing paper overlap between the 3 RQ methodology labels.

        Args:
            master        : Screened DataFrame with binary method columns.
            method_labels : Exactly 3 methodology label column names.
            out_dir       : Directory to save the PNG.
            filename      : Output filename.
        """
        if len(method_labels) != 3:
            raise ValueError("venn_rq_overlap requires exactly 3 method labels.")

        a, b, c = method_labels
        df = master.copy()

        # Compute the 7 Venn regions
        A  = df[a].astype(bool)
        B  = df[b].astype(bool)
        C  = df[c].astype(bool)

        only_a   = int((A  & ~B & ~C).sum())
        only_b   = int((~A &  B & ~C).sum())
        only_c   = int((~A & ~B &  C).sum())
        ab_only  = int((A  &  B & ~C).sum())
        ac_only  = int((A  & ~B &  C).sum())
        bc_only  = int((~A &  B &  C).sum())
        abc      = int((A  &  B &  C).sum())

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")

        colors = ["#4C72B0", "#DD8452", "#55A868"]
        cx = [-0.38,  0.38,  0.0]
        cy = [ 0.22,  0.22, -0.28]
        r  = 0.60

        for i in range(3):
            circle = plt.Circle((cx[i], cy[i]), r,
                                color=colors[i], alpha=0.28,
                                linewidth=2, edgecolor=colors[i])
            ax.add_patch(circle)

        # Short labels on circle edges
        short = [lb.split()[0] for lb in method_labels]
        label_pos = [(-0.82, 0.72), (0.82, 0.72), (0.0, -1.05)]
        for i, (lx, ly) in enumerate(label_pos):
            ax.text(lx, ly, short[i], ha="center", va="center",
                    fontsize=10, fontweight="bold", color=colors[i])

        # Annotate each region — counts only, all black
        kw = dict(ha="center", va="center", fontsize=12, fontweight="bold", color="black")
        ax.text(-0.72,  0.22, str(only_a),  **kw)
        ax.text( 0.72,  0.22, str(only_b),  **kw)
        ax.text( 0.00, -0.72, str(only_c),  **kw)
        ax.text( 0.00,  0.48, str(ab_only), **kw)
        ax.text(-0.40, -0.22, str(ac_only), **kw)
        ax.text( 0.40, -0.22, str(bc_only), **kw)
        ax.text( 0.00,  0.06, str(abc),     **kw)

        # Full labels as legend below
        legend_patches = [
            mpatches.Patch(color=colors[i], alpha=0.6, label=method_labels[i])
            for i in range(3)
        ]
        ax.legend(handles=legend_patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=1,
                  fontsize=9, frameon=False)

        total = only_a + only_b + only_c + ab_only + ac_only + bc_only + abc
        ax.set_title(
            f"RQ Overlap — Title-Screened Papers (n = {total})",
            fontsize=13, fontweight="bold", pad=14,
        )
        plt.tight_layout()

        if out_dir:
            out_path = Path(out_dir) / filename
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"    Plot saved: {filename}")