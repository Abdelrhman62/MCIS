import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

class CVManager:
    def __init__(self, n_splits=5, random_state=42, output_dir="./data/folds"):
        self.n_splits = n_splits
        self.random_state = random_state
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

    def create_folds(self, df, text_col='text', label_col='label'):
        """
        Creates 5 folds and saves them to disk to ensure consistency across experiments.
        """
        print(f"Creating {self.n_splits}-fold cross-validation splits...")
        
        # Note: If this is multi-label, you should use IterativeStratification from skmultilearn instead.
        # This assumes multi-class (e.g., Primary Diagnosis).
        X = df[text_col]
        y = df[label_col]
        
        fold_records = []
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            # Save folds to disk for reproducibility
            train_path = os.path.join(self.output_dir, f"fold_{fold}_train.csv")
            val_path = os.path.join(self.output_dir, f"fold_{fold}_val.csv")
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            
            fold_records.append({'fold': fold, 'train_size': len(train_df), 'val_size': len(val_df)})
            print(f"Fold {fold}: Train={len(train_df)} | Val={len(val_df)}")
            
        return fold_records

    def load_fold(self, fold_idx):
        """Loads a specific fold for training/evaluation."""
        train_path = os.path.join(self.output_dir, f"fold_{fold_idx}_train.csv")
        val_path = os.path.join(self.output_dir, f"fold_{fold_idx}_val.csv")
        return pd.read_csv(train_path), pd.read_csv(val_path)