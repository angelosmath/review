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

        # Υπολογίζει το σύνολο ανά έτος (absolute)
        yearly = df.groupby("Year")[themes].sum().sort_index()

        # Υπολογίζει το ποσοστό (ανά έτος) για να δείξει το "μερίδιο αγοράς" της κάθε τεχνολογίας
        yearly_totals = df.groupby("Year").size()
        yearly_normalized = np.log10(yearly + 1)

        # Μετά κάνεις ax.plot() χρησιμοποιώντας το yearly_normalized

        fig, ax = plt.subplots(figsize=(12, 6))
        markers = ["o", "s", "D", "^", "v"]
        for idx, theme in enumerate(themes):
            ax.plot( 
                yearly_normalized.index, yearly_normalized[theme],
                marker=markers[idx % len(markers)],
                linewidth=2.5, markersize=7,
                label=theme,
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Paper Count", fontsize=11)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        if out_dir:
            out_path = Path(out_dir) / filename
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"    Plot saved: {filename}")