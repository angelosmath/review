import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

class LiteraturePlots:
    @staticmethod
    def identification_summary(master: pd.DataFrame,out_dir: str = None):
        total = len(master)

        pubmed = int((master["Source"] == "PubMed").sum())
        scopus = int((master["Source"] == "Scopus").sum())
        ieee   = int((master["Source"] == "IEEE").sum())

        # Paper-level counts using dedup_key
        if "dedup_key" not in master.columns:
            raise ValueError("dedup_key is missing. Run _deduplicate() first.")

        dedup_papers = int(master["dedup_key"].nunique())
        duplicates_removed = int(total - dedup_papers)

        # Exclusions at paper level (unique keys that are flagged)
        if "flag_review_conf" in master.columns:
            excluded_papers = int(master.loc[master["flag_review_conf"], "dedup_key"].nunique())
        else:
            excluded_papers = 0

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

        # (κρατάω τα χρώματά σου)
        colors = [
            "#4C72B0", "#A1ADC2", "#373B42",
            "#2ecc71",
            "#e74c3c",
            "#27ae60",
            "#f39c12",
            "#16a085"
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
            mpatches.Patch(color="#4C72B0", label="Records by source"),
            mpatches.Patch(color="#2ecc71", label="Total records (raw)"),
            mpatches.Patch(color="#e74c3c", label="Duplicates removed (raw→dedup)"),
            mpatches.Patch(color="#27ae60", label="Unique papers (dedup)"),
            mpatches.Patch(color="#f39c12", label="Excluded papers (review+conf)"),
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
        plt.show()

    # @staticmethod
    # def thematic_heatmap(...)        
    
    # @staticmethod
    # def temporal_trends(...)          
    
    # @staticmethod
    # def cooccurrence_matrix(...)      
    
    # @staticmethod
    # def cancer_type_distribution(...) 