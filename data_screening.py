from transformers import pipeline
import pandas as pd
import torch

class DataScreening:
    def __init__(self, master_df: pd.DataFrame):

        df_no_reviews = master_df[~master_df["flag_review_conf"]].copy()
        self.df = df_no_reviews.drop_duplicates(subset=['dedup_key'], keep='first').copy()
        
        print(f"[NLP] Final papers for analysis: {len(self.df)}") 
        
        print("[NLP] Loading Bio-Inference Model...")
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("zero-shot-classification", 
                  model="facebook/bart-large-mnli", 
                  device=device)


    
    def intelligent_thematic_tagging(self, candidate_labels, threshold=0.35):
        """
        Αναλύει κάθε τίτλο και αναθέτει πιθανότητες στα θέματα που ορίζεις.
        """
        print(f"[NLP] Analyzing {len(self.df)} titles...")
        titles = self.df["Title"].tolist()
        
        # Classification
        results = self.classifier(
            titles, 
            candidate_labels, 
            multi_label=True,
            #hypothesis_template="This scientific paper is about {}."
        )

        print(f"\n[DEBUG] Το 1ο paper πήρε τα εξής σκορ:")
        for i in range(3): # Τυπώνει τα 3 κορυφαία
            print(f"  - {results[0]['labels'][i]}: {results[0]['scores'][i]:.3f}")
        print("-" * 40)
        
        for label in candidate_labels:
            self.df[label] = [
                1 if label in res['labels'] and res['scores'][res['labels'].index(label)] > threshold 
                else 0 for res in results
            ]
        
        return self.df