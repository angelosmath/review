

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


class LiteraturePlots:
    """
    Collection of static plot methods.
    """

    @staticmethod
    def identification_summary(combined: pd.DataFrame, master_unique: pd.DataFrame):
        """
        Creates a bar chart summarizing the identification step.

        Parameters
        ----------
        combined      : full combined DataFrame (with duplicates)
        master_unique : deduplicated DataFrame (output of LiteratureIdentification)
        """
        total      = len(combined)
        unique     = len(master_unique)
        duplicates = total - unique
        reviews    = int(master_unique["flag_review"].sum())

        labels = [
            "PubMed",
            "Scopus",
            "IEEE",
            "Total\n(raw)",
            "Unique\n(dedup)",
            "Duplicates\nremoved",
            "Review\nflagged"
        ]
        values = [
            len(combined[combined["Source"] == "PubMed"]),
            len(combined[combined["Source"] == "Scopus"]),
            len(combined[combined["Source"] == "IEEE"]),
            total,
            unique,
            duplicates,
            reviews
        ]
        colors = [
            "#4C72B0", "#4C72B0", "#4C72B0",  # source bars — blue
            "#2ecc71",                          # total raw — green
            "#27ae60",                          # unique — dark green
            "#e74c3c",                          # duplicates — red
            "#f39c12"                           # reviews — orange
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)

        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(val),
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

        # Legend
        legend_patches = [
            mpatches.Patch(color="#4C72B0", label="Records per source"),
            mpatches.Patch(color="#2ecc71", label="Combined total"),
            mpatches.Patch(color="#27ae60", label="Unique after dedup"),
            mpatches.Patch(color="#e74c3c", label="Duplicates removed"),
            mpatches.Patch(color="#f39c12", label="Review articles flagged"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

        ax.set_title("PRISMA — Identification Summary", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Number of Records", fontsize=11)
        ax.set_ylim(0, max(values) * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig("step01_identification_summary.png", dpi=300)
        plt.show()
        print("    Saved: step01_identification_summary.png")


    






    @staticmethod
    def thematic_heatmap(...)        
    
    @staticmethod
    def temporal_trends(...)          
    
    @staticmethod
    def cooccurrence_matrix(...)      
    
    @staticmethod
    def cancer_type_distribution(...) 