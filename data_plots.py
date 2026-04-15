import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
import seaborn as sns
import re
import numpy as np


class LiteraturePlots:
    @staticmethod
    def identification_summary(master: pd.DataFrame,out_dir: Path | None = None):
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
            # Sources (match the first three bars exactly)
            mpatches.Patch(color="#4C72B0", label="PubMed records"),
            mpatches.Patch(color="#A1ADC2", label="Scopus records"),
            mpatches.Patch(color="#373B42", label="IEEE records"),

            # Pipeline steps (match remaining bars)
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
        #plt.show()



    @staticmethod
    def thematic_heatmap(master: pd.DataFrame, themes: dict, out_dir: Path | None = None):
        
        # Δεν κάνουμε ξανά drop_duplicates γιατί έχει γίνει σωστά στο AI Screening
        df = master.copy()

        # Παίρνουμε τα ονόματα των στηλών που έχει ήδη υπολογίσει το AI (από τα keys του dict)
        theme_names = list(themes.keys())

        # Υπολογισμός Co-occurrence Matrix χρησιμοποιώντας ΑΠΕΥΘΕΙΑΣ τις AI στήλες
        matrix = pd.DataFrame(index=theme_names, columns=theme_names, dtype=int)
        for t1 in theme_names:
            for t2 in theme_names:
                # Κάνουμε απλή καταμέτρηση όπου και τα δύο (t1, t2) είναι 1
                matrix.loc[t1, t2] = ((df[t1] == 1) & (df[t2] == 1)).sum()

        matrix_int = matrix.astype(int)
        
        # Χρησιμοποιούμε "Reds" για να φανεί έντονα το λευκό κενό των Agents
        plt.figure(figsize=(14, 12))
        sns.heatmap(matrix_int, annot=True, fmt="d", cmap="Reds", square=True,
                    cbar_kws={'label': 'Paper Count'},
                    annot_kws={"size": 14, "weight": "bold"})
        
        plt.title("Thematic Co-occurrence Heatmap", fontsize=16, fontweight="bold", pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()

        if out_dir:
            plt.savefig(Path(out_dir) / "step02_thematic_heatmap.png", dpi=300)
        #plt.show()



    # @staticmethod
    # def thematic_heatmap(...)        
    
    # @staticmethod
    # def temporal_trends(...)          
    
    # @staticmethod
    # def cooccurrence_matrix(...)      
    
    # @staticmethod
    # def cancer_type_distribution(...) 