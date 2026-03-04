import re
import pandas as pd
from data_analysis import LiteraturePlots
from pathlib import Path


# ============================================================
# CLASS: LiteratureIdentification
# ============================================================

class LiteratureIdentification:

    def __init__(self, pubmed_path: str, scopus_path: str, ieee_path: str):
        self.pubmed_path  = pubmed_path
        self.scopus_path  = scopus_path
        self.ieee_path    = ieee_path
        self.master = None  
        self.OUT_DIR = Path(__file__).resolve().parent.parent / "output" 
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        self._print_summary()

        LiteraturePlots.identification_summary(self.master, out_dir=self.OUT_DIR)

        return self.master

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
            "Title":   self.col(pub, "Title"),
            "Authors": self.col(pub, "Authors"),
            "Year":    self.col(pub, "Publication Year"),
            "DOI":     self.col(pub, "DOI"),
            "Source":  "PubMed"
        })

        scopus_df = pd.DataFrame({
            "Title":   self.col(scopus, "Title"),
            "Authors": self.col(scopus, "Author full names", default=self.col(scopus, "Authors")),
            "Year":    self.col(scopus, "Year"),
            "DOI":     self.col(scopus, "DOI"),
            "Source":  "Scopus"
        })

        ieee_df = pd.DataFrame({
            "Title":   self.col(ieee, "Document Title", default=self.col(ieee, "Title")),
            "Authors": self.col(ieee, "Authors"),
            "Year":    self.col(ieee, "Publication Year"),
            "DOI":     self.col(ieee, "DOI"),
            "Source":  "IEEE"
        })


        print("Columns PubMed:", list(pub.columns))
        print("Columns Scopus:", list(scopus.columns))
        print("Columns IEEE:", list(ieee.columns))

        self.master = pd.concat(
            [pub_df, scopus_df, ieee_df], ignore_index=True
        )

        print(f"    PubMed : {len(pub_df)} records")
        print(f"    Scopus : {len(scopus_df)} records")
        print(f"    IEEE   : {len(ieee_df)} records")
        print(f"    Total  : {len(self.master)} records")

    # ----------------------------------------------------------
    # STEP 2: Normalize text fields for comparison
    # ----------------------------------------------------------
    def _normalize(self):        #
        """Create clean/normalized versions of DOI, Title and Year."""
        print("\n[2/5] Normalizing fields...")

        self.master["DOI_norm"]   = self.master["DOI"].apply(self.norm_doi)
        self.master["Title_norm"] = self.master["Title"].apply(self.norm_title)
        self.master["Year"]       = (
            self.master["Year"]
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
            .fillna("")
        )

    # ----------------------------------------------------------
    # STEP 3: Deduplicate
    # ----------------------------------------------------------
    def _deduplicate(self):
        print("\n[3/5] Flagging duplicates (no deletion)...")

        df = self.master

        no_doi    = df["DOI_norm"].eq("")
        has_title = df["Title_norm"].ne("")
        has_year  = df["Year"].ne("")

        # Start with DOI-based key
        df["dedup_key"] = df["DOI_norm"]

        # 1) No DOI but Title+Year
        mask_ty = no_doi & has_title & has_year
        df.loc[mask_ty, "dedup_key"] = (
            "TY:" + df.loc[mask_ty, "Title_norm"] + "|Y:" + df.loc[mask_ty, "Year"]
        )

        # 2) No DOI, Title exists but no Year → Title only
        mask_t = no_doi & has_title & (~has_year)
        df.loc[mask_t, "dedup_key"] = "T:" + df.loc[mask_t, "Title_norm"]

        # 3) No DOI and empty Title → unique ROW key
        mask_row = no_doi & (~has_title)
        df.loc[mask_row, "dedup_key"] = df.loc[mask_row].index.map(lambda i: f"ROW:{i}")

        # group size + duplicate flag
        df["dup_group_size"] = df.groupby("dedup_key")["dedup_key"].transform("size")
        df["flag_duplicate"] = df["dup_group_size"].gt(1)

        print("    QA — Missing DOI   :", int(no_doi.sum()))
        print("    QA — Missing Title :", int((~has_title).sum()))
        print("    QA — Missing Year  :", int((~has_year).sum()))
        print("    Duplicate rows     :", int(df["flag_duplicate"].sum()))


    # ----------------------------------------------------------
    # STEP 4: Flag review articles
    # ----------------------------------------------------------
    def _flag_reviews(self):
        print("\n[4/5] Flagging review/conference papers...")
        self.master["flag_review_conf"] = self.master["Title"].apply(self._is_review)

    # ----------------------------------------------------------
    # STEP 5: Save outputs
    # ----------------------------------------------------------
    def _save(self):
        print("\n[5/5] Saving outputs...")
        out_path = self.OUT_DIR / "step01_master_flagged.csv"
        self.master.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"    Saved: {out_path}")

    # ----------------------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------------------
    def _print_summary(self):
        total = len(self.master)
        dedup_papers = self.master["dedup_key"].nunique()
        dupes_removed = len(self.master) - dedup_papers 
        revconf = int(self.master.loc[self.master["flag_review_conf"], "dedup_key"].nunique())

        print("\n" + "=" * 50)
        print(" IDENTIFICATION SUMMARY")
        print("=" * 50)
        print(f"  Total records loaded        : {total}")
        print(f"  duplicate    : {dupes_removed}")
        print(f"  Review/Conf flagged (rows)  : {revconf}")
        print("=" * 50)
        
        
    
    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------


    @staticmethod 
    def norm_text(x) -> str:
        if pd.isna(x):
            return ""
        x = str(x).lower().strip()
        x = re.sub(r"\s+", " ", x)
        return x
    
    @staticmethod
    def norm_doi(x) -> str:
        if pd.isna(x):
            return ""
        x = str(x).strip().lower()
        x = re.sub(r"^doi:\s*", "", x)
        x = re.sub(r"^https?://(dx\.)?doi\.org/", "", x)
        x = re.sub(r"\s+", "", x)
        x = x.rstrip(".,;")                  # κόψε trailing punctuation
        return x

    @staticmethod
    def norm_title(x) -> str:
        x = LiteratureIdentification.norm_text(x)   # <-- ΟΧΙ self
        x = re.sub(r"[–—−]", "-", x)
        x = re.sub(r"[^\w\s\-]", "", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x
        
    @staticmethod
    def col(df: pd.DataFrame, name: str, default=""):
        """Safe column getter: returns df[name] if exists else default (scalar or Series)."""
        if name in df.columns:
            return df[name]
        if not isinstance(default, pd.Series):
            return pd.Series([default] * len(df), index=df.index)
        return default
    
    
    def _is_review(self, title: str) -> bool:
        t = LiteratureIdentification.norm_title(title)

        # strong phrases
        strong = [
            "systematic review", "scoping review", "literature review",
            "narrative review", "umbrella review", "meta-analysis",
            "meta analysis", "survey", "state of the art", "proceedings",
            "symposium", "workshop", "congress", "conference"
        ]
        if any(s in t for s in strong):
            return True

        # whole-word review
        return re.search(r"\breview\b", t) is not None
    
# ============================================================
# ENTRY POINT 
# ============================================================
if __name__ == "__main__":
    identifier = LiteratureIdentification(
        pubmed_path = "/home/amath/Desktop/review_v1/data/csv-KnowledgeG-set.csv",
        scopus_path = "/home/amath/Desktop/review_v1/data/scopus_export_Feb 25-2026_d93c044b-5fec-4da5-8d30-ad7cb2501782.csv",
        ieee_path   = "/home/amath/Desktop/review_v1/data/export2026.02.25-06.45.35.csv"
    )
    master = identifier.run()
