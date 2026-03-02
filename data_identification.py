import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data_analysis import LiteraturePlots

# ============================================================
# CLASS: LiteratureIdentification
# ============================================================

class LiteratureIdentification:
    """
    Loads and combines databases CSV exports into a
    single deduplicated DataFrame.

    Parameters
    ----------
    pubmed_path  : str  
    scopus_path  : str  
    ieee_path    : str  

    Usage
    -----
    identifier = LiteratureIdentification(
        pubmed_path  = "csv-KnowledgeG-set.csv",
        scopus_path  = "scopus_export_....csv",
        ieee_path    = "export2026....csv"
    )
    master = identifier.run()   # returns the final DataFrame
    """

    # Source loading priority for deduplication
    # When the same paper appears in multiple databases,
    # we keep the record from the highest-priority source.
    SOURCE_PRIORITY = {"PubMed": 0, "Scopus": 1, "IEEE": 2}

    # Keywords that indicate a paper is itself a review article.
    # We flag these but do NOT remove them — the decision to
    # include or exclude review articles is made in data_screening.py
    REVIEW_KEYWORDS = [
        "systematic review",
        "scoping review",
        "literature review",
        "narrative review",
        "umbrella review",
        "meta-analysis",
        "meta analysis",
        "survey",
        "state of the art",
        "review"
    ]

    def __init__(self, pubmed_path: str, scopus_path: str, ieee_path: str):
        self.pubmed_path  = pubmed_path
        self.scopus_path  = scopus_path
        self.ieee_path    = ieee_path
        self.combined     = None   # all records including duplicates
        self.master_unique = None  # deduplicated records

    # ----------------------------------------------------------
    # PUBLIC METHOD: run everything in sequence
    # ----------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute full identification pipeline and return master_unique."""
        print("\n" + "=" * 50)
        print(" LITERATURE IDENTIFICATION PIPELINE")
        print("=" * 50)

        self._load()
        self._normalize()
        self._deduplicate()
        self._flag_reviews()
        self._save()
        self._plot()
        self._print_summary()

        LiteraturePlots.identification_summary(self.combined, self.master_unique)


        return self.master_unique

    # ----------------------------------------------------------
    # STEP 1: Load each CSV and map to common schema
    # ----------------------------------------------------------
    def _load(self):
        """Load the 3 CSVs and standardize column names."""
        print("\n[1/5] Loading CSV files...")

        pub    = pd.read_csv(self.pubmed_path)
        scopus = pd.read_csv(self.scopus_path)
        ieee   = pd.read_csv(self.ieee_path)

        pub_df = pd.DataFrame({
            "Title"   : pub["Title"],
            "Authors" : pub["Authors"],
            "Year"    : pub["Publication Year"],
            "Journal" : pub["Journal/Book"],
            "DOI"     : pub["DOI"],
            "Source"  : "PubMed"
        })

        scopus_df = pd.DataFrame({
            "Title"   : scopus["Title"],
            "Authors" : scopus["Author full names"],
            "Year"    : scopus["Year"],
            "Journal" : scopus["Source title"],
            "DOI"     : scopus["DOI"],
            "Source"  : "Scopus"
        })

        ieee_df = pd.DataFrame({
            "Title"   : ieee["Document Title"],
            "Authors" : ieee["Authors"],
            "Year"    : ieee["Publication Year"],
            "Journal" : ieee["Publication Title"],
            "DOI"     : ieee["DOI"],
            "Source"  : "IEEE"
        })

        self.combined = pd.concat(
            [pub_df, scopus_df, ieee_df], ignore_index=True
        )

        print(f"    PubMed : {len(pub_df)} records")
        print(f"    Scopus : {len(scopus_df)} records")
        print(f"    IEEE   : {len(ieee_df)} records")
        print(f"    Total  : {len(self.combined)} records")

    # ----------------------------------------------------------
    # STEP 2: Normalize text fields for comparison
    # ----------------------------------------------------------
    def _normalize(self):
        """Create clean/normalized versions of DOI, Title and Year."""
        print("\n[2/5] Normalizing fields...")

        self.combined["DOI_norm"]   = self.combined["DOI"].apply(self._norm_doi)
        self.combined["Title_norm"] = self.combined["Title"].apply(self._norm_title)
        self.combined["Year"]       = (
            self.combined["Year"]
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
            .fillna("")
        )

    # ----------------------------------------------------------
    # STEP 3: Deduplicate
    # ----------------------------------------------------------
    def _deduplicate(self):
        """
        Build a dedup_key for each record:
          - If DOI exists → use normalized DOI
          - If no DOI    → use normalized Title + Year
        Then keep the best record per group (by source priority).
        """
        print("\n[3/5] Deduplicating...")

        # Build dedup key
        self.combined["dedup_key"] = self.combined["DOI_norm"]
        no_doi = self.combined["dedup_key"].eq("")
        self.combined.loc[no_doi, "dedup_key"] = self.combined.loc[no_doi].apply(
            lambda r: f"T:{r['Title_norm']}|Y:{r['Year']}", axis=1
        )

        # Flag duplicates
        self.combined["dup_group_size"] = (
            self.combined.groupby("dedup_key")["dedup_key"].transform("size")
        )
        self.combined["flag_duplicate"] = self.combined["dup_group_size"] > 1

        # Keep best record per group
        self.combined["source_rank"] = self.combined["Source"].map(self.SOURCE_PRIORITY)
        sorted_df = self.combined.sort_values(["dedup_key", "source_rank"])
        self.master_unique = (
            sorted_df.drop_duplicates(subset="dedup_key", keep="first").copy()
        )
        self.master_unique = self.master_unique.drop(columns=["source_rank"])

    # ----------------------------------------------------------
    # STEP 4: Flag review articles
    # ----------------------------------------------------------
    def _flag_reviews(self):
        """
        Flag papers whose title suggests they are review articles.
        This does NOT remove them.
        """
        print("\n[4/5] Flagging review articles...")

        self.master_unique["flag_review"] = (
            self.master_unique["Title"]
            .apply(lambda t: self._is_review(t))
        )

    # ----------------------------------------------------------
    # STEP 5: Save outputs
    # ----------------------------------------------------------
    def _save(self):
        """Save the deduplicated master to CSV."""
        print("\n[5/5] Saving outputs...")
        self.master_unique.to_csv(
            "step01_master_unique.csv", index=False, encoding="utf-8-sig"
        )
        print("    Saved: step01_master_unique.csv")

    # ----------------------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------------------
    def _print_summary(self):
        total      = len(self.combined)
        unique     = len(self.master_unique)
        duplicates = total - unique
        reviews    = int(self.master_unique["flag_review"].sum())

        print("\n" + "=" * 50)
        print(" IDENTIFICATION SUMMARY")
        print("=" * 50)
        print(f"  Total records loaded       : {total}")
        print(f"  Unique papers after dedup  : {unique}")
        print(f"  Duplicates removed         : {duplicates}")
        print(f"  Review articles flagged    : {reviews}")
        print("=" * 50)
        print("\nReady for Step 2: data_screening.py")

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------
    @staticmethod
    def _norm_doi(x):
        if pd.isna(x):
            return ""
        x = str(x).strip().lower()
        for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
            x = x.replace(prefix, "")
        return re.sub(r"\s+", "", x)

    @staticmethod
    def _norm_title(x):
        if pd.isna(x):
            return ""
        x = str(x).lower().strip()
        x = re.sub(r"[^\w\s]", " ", x)
        return re.sub(r"\s+", " ", x).strip()

    def _is_review(self, title):
        t = self._norm_title(title)
        return any(kw in t for kw in self.REVIEW_KEYWORDS)
    




# ============================================================
# ENTRY POINT 
# ============================================================
if __name__ == "__main__":
    identifier = LiteratureIdentification(
        pubmed_path = "csv-KnowledgeG-set.csv",
        scopus_path = "scopus_export_Feb_25-2026_d93c044b-5fec-4da5-8d30-ad7cb2501782.csv",
        ieee_path   = "export2026_02_25-06_45_35.csv"
    )
    master = identifier.run()


