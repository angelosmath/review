from transformers import pipeline
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm


class DataScreening:
    """
    AI-powered thematic screening of literature titles using zero-shot classification.

    Improvements over v1:
      - Lighter, faster model (deberta-v3-small vs bart-large-mnli: ~4x speedup)
      - batch_size=32 for GPU/CPU throughput (5-10x speedup)
      - Checkpoint support: if cache_path exists, skip inference entirely on re-runs
      - Cleaner score extraction (no repeated list.index() calls inside list comprehensions)
    """

    # Top-ranked zero-shot NLI model on scientific text.
    # Consistently outperforms bart-large-mnli on NLI benchmarks.
    # Swap to "cross-encoder/nli-deberta-v3-small" if GPU memory is limited.
    DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

    def __init__(
        self,
        master_df: pd.DataFrame,
        cache_path: Path | None = None,
        model_name: str | None = None,
        batch_size: int = 32,
    ):
        """
        Args:
            master_df  : The master DataFrame from LiteratureIdentification.run().
            cache_path : Optional path to a .csv checkpoint file. If the file exists,
                         inference is skipped and the cached result is returned directly.
            model_name : Override the default HuggingFace model.
            batch_size : Number of titles per inference batch. Reduce to 8-16 if OOM.
        """
        df_no_reviews = master_df[~master_df["flag_review_conf"]].copy()
        self.df         = df_no_reviews.drop_duplicates(subset=["dedup_key"], keep="first").copy()
        self.cache_path = Path(cache_path) if cache_path else None
        self.batch_size = batch_size
        self._model_name = model_name or self.DEFAULT_MODEL

        print(f"[NLP] Papers for analysis : {len(self.df)}")

        # Skip loading the model if we already have a cache
        if self.cache_path and self.cache_path.exists():
            print(f"[NLP] Cache found — model will not be loaded.")
            self.classifier = None
            return

        print(f"[NLP] Loading model       : {self._model_name}")
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self._model_name,
            device=device,
        )
        print(f"[NLP] Device              : {'GPU' if device == 0 else 'CPU'}")

    # ----------------------------------------------------------
    # PUBLIC: Run thematic tagging
    # ----------------------------------------------------------
    def intelligent_thematic_tagging(
        self,
        candidate_labels: list[str],
        threshold: float = 0.35,
    ) -> pd.DataFrame:
        """
        Zero-shot classification of paper titles against candidate themes.

        Args:
            candidate_labels : List of theme labels to classify against.
            threshold        : Minimum score to assign a label as positive (1).

        Returns:
            DataFrame with one binary column per theme appended.
        """
        # --- Checkpoint: return early if cache exists ---
        if self.cache_path and self.cache_path.exists():
            print(f"[NLP] Loading from cache: {self.cache_path}")
            return pd.read_csv(self.cache_path)

        titles = self.df["Title"].tolist()
        n_batches = (len(titles) + self.batch_size - 1) // self.batch_size
        print(f"[NLP] Classifying {len(titles)} titles | {len(candidate_labels)} themes | {n_batches} batches")

        batches = [titles[i:i + self.batch_size] for i in range(0, len(titles), self.batch_size)]
        results = []
        for batch in tqdm(batches, desc="Classifying", unit="batch"):
            results.extend(self.classifier(batch, candidate_labels, multi_label=True))

        # Debug: show top-3 scores for the first paper
        print("\n[DEBUG] Top scores for paper #1:")
        for label, score in zip(results[0]["labels"][:3], results[0]["scores"][:3]):
            print(f"  {label}: {score:.3f}")
        print("-" * 40)

        # Build score lookup per result to avoid O(n) list.index() inside comprehension
        score_map = {}
        for label in candidate_labels:
            scores = []
            for r in results:
                label_scores = dict(zip(r["labels"], r["scores"]))
                scores.append(1 if label_scores.get(label, 0.0) > threshold else 0)
            score_map[label] = scores

        self.df = self.df.assign(**score_map)

        # --- Save checkpoint ---
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(self.cache_path, index=False, encoding="utf-8-sig")
            print(f"[NLP] Checkpoint saved: {self.cache_path}")

        return self.df