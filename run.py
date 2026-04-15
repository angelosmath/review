import pandas as pd
from pathlib import Path
from data_identification import LiteratureIdentification
from data_screening import DataScreening
from data_plots import LiteraturePlots

def main():
    print("\n>>> STARTING STEP 01: DATA IDENTIFICATION")
    BASE_DIR = Path("/home/amath/Desktop/review_v1")
    DATA_DIR = BASE_DIR / "data"
    OUT_DIR  = BASE_DIR / "output"

    # --- STEP 01 ---
    identifier = LiteratureIdentification(
        pubmed_path = str(DATA_DIR / "csv-KnowledgeG-set.csv"),
        scopus_path = str(DATA_DIR / "scopus_export_Feb 25-2026_d93c044b-5fec-4da5-8d30-ad7cb2501782.csv"),
        ieee_path   = str(DATA_DIR / "export2026.02.25-06.45.35.csv")
    )
    master = identifier.run() 

    # --- STEP 02: INTELLIGENT SCREENING ---
    print("\n>>> STARTING STEP 02: AI THEMATIC ANALYSIS")
    screener = DataScreening(master)
    

    
    model_labels = [
        "Knowledge Graphs",
        "Foundation Models",
        "Autonomous Agents",
        "Mechanistic Simulations",
        "Graph Neural Networks",
        "Therapy Response Prediction"  
    ]  

    master_tagged = screener.intelligent_thematic_tagging(model_labels, threshold=0.25)

    
    rename_mapping = {
        "Knowledge Graphs": "Knowledge Graphs & Clinical Data",
        "Graph Neural Networks": "Static Graph Models (GNNs)",           # Τονίζουμε ότι το τρέχον SOTA είναι στατικό
        "Therapy Response Prediction": "Therapy Response & Prognosis",   # Ίδιο με τον τίτλο του TRIAGE!
        "Foundation Models": "Multimodal Foundation Models",             # Το VPH K-FM σου
        "Mechanistic Simulations": "Mechanistic Tumor Simulations",      # Τα συνθετικά σου δεδομένα
        "Autonomous Agents": "Agentic AI Decision Support"               # Η απόλυτη καινοτομία σου
    }

    master_tagged = master_tagged.rename(columns=rename_mapping)

    # Αποθήκευση
    csv_path = OUT_DIR / "step02_master_with_ai_tags.csv"
    master_tagged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Dataset with AI tags saved to: {csv_path}")

    # --- STEP 03: VISUALIZATION ---
    print("\n>>> STARTING STEP 03: VISUALIZATION")
    
    fancy_labels = list(rename_mapping.values())
    custom_themes = {label: [label] for label in fancy_labels}
    
    LiteraturePlots.thematic_heatmap(master_tagged, themes=custom_themes, out_dir=OUT_DIR)
    
    print("\nPipeline execution finished successfully!")

if __name__ == "__main__":
    main()