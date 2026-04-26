import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

class MACS_Metrics:
    def __init__(self, y_true, y_pred, y_prob=None, classes=None):
        """
        y_true: Ground truth labels
        y_pred: Model predictions
        y_prob: Prediction probabilities (optional, for AUC/ROC later)
        classes: List of class names (optional)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.classes = classes

    def evaluate_all(self):
        """Calculates and prints all required MACS metrics."""
        print("=== MACS Model Evaluation Results ===")
        
        # Accuracy
        acc = accuracy_score(self.y_true, self.y_pred)
        print(f"Overall Accuracy:  {acc:.4f}")
        
        # Micro metrics (Calculates metrics globally by counting the total true positives, false negatives and false positives)
        micro_f1 = f1_score(self.y_true, self.y_pred, average='micro', zero_division=0)
        print(f"Micro F1-Score:    {micro_f1:.4f}")
        
        # Macro metrics (Calculates metrics for each label, and finds their unweighted mean. Does not take label imbalance into account)
        macro_f1 = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        print(f"Macro F1-Score:    {macro_f1:.4f}")
        
        # Weighted metrics (Crucial for class imbalance - finds F1 for each class and averages them based on support)
        weighted_f1 = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        
        weighted_prec = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        weighted_rec = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        print(f"Weighted Precision:{weighted_prec:.4f}")
        print(f"Weighted Recall:   {weighted_rec:.4f}")
        print("=====================================\n")
        
        return {
            'accuracy': acc,
            'f1_micro': micro_f1,
            'f1_macro': macro_f1,
            'f1_weighted': weighted_f1,
            'precision_weighted': weighted_prec,
            'recall_weighted': weighted_rec
        }

    def print_classification_report(self):
        """Prints the detailed scikit-learn classification report."""
        print(classification_report(self.y_true, self.y_pred, target_names=self.classes, zero_division=0))