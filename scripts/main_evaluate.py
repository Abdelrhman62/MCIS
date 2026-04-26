import torch
import pandas as pd
from cv_manager import CVManager
from metrics import MACS_Metrics
from unk_scanner import scan_unknown_tokens

def evaluate_b0_baseline(model, df_path, text_col='clinical_text', label_col='icd_code'):
    """
    Runs the full Week 1 Evaluation Task: 5-Fold CV + Metrics + UNK Scanning
    """
    print("--- Starting MACS Evaluation Pipeline ---")
    
    # 1. Load Data & Initialize Fold Manager
    df = pd.read_csv(df_path)
    cv = CVManager(n_splits=5)
    folds = cv.create_folds(df, text_col=text_col, label_col=label_col)
    
    all_fold_metrics = []
    
    # 2. Iterate through all 5 folds
    for fold_idx in range(5):
        print(f"\nEvaluating Fold {fold_idx + 1}/5...")
        train_df, val_df = cv.load_fold(fold_idx)
        
        # Run Malak's UNK Scanner to see if test data has new vocabulary
        scan_unknown_tokens(train_df[text_col], val_df[text_col])
        
        # NOTE: Abdelrhman's training code happens here. 
        # model.train(train_df) 
        
        # 3. Inference (Predicting on the validation fold)
        model.eval()
        y_true = val_df[label_col].tolist()
        y_pred = []
        
        with torch.no_grad():
            for text in val_df[text_col]:
                # Assuming model returns a tensor of class indices
                prediction = model.predict(text) 
                y_pred.append(prediction.item())
        
        # 4. Calculate MACS Metrics using your Day 4 script
        evaluator = MACS_Metrics(y_true, y_pred)
        fold_results = evaluator.evaluate_all()
        all_fold_metrics.append(fold_results)
    
    # 5. Calculate Final Averages across all 5 folds
    print("\n=== FINAL 5-FOLD CROSS VALIDATION RESULTS ===")
    avg_f1_macro = sum(f['f1_macro'] for f in all_fold_metrics) / 5
    avg_f1_weighted = sum(f['f1_weighted'] for f in all_fold_metrics) / 5
    avg_acc = sum(f['accuracy'] for f in all_fold_metrics) / 5
    
    print(f"Average Macro F1:    {avg_f1_macro:.4f}")
    print(f"Average Weighted F1: {avg_f1_weighted:.4f}")
    print(f"Average Accuracy:    {avg_acc:.4f}")
    print("---------------------------------------------")

# To run this when Abdelrhman gives you the model:
# from models import Abdelrhman_PathologyBERT
# model = Abdelrhman_PathologyBERT()
# evaluate_b0_baseline(model, "baheya_cleaned_dataset.csv")