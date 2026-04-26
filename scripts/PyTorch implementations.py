import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Standard Focal Loss for highly imbalanced medical datasets.
    Down-weights easy examples (like common IDC cancers) to focus on hard, rare examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class AsymmetricPolynomialLoss(nn.Module):
    """
    APL (Asymmetric Polynomial Loss) based on the Niemi 2025 Architecture.
    Specifically designed to be robust against NOISY LABELS in pathology reports.
    It applies different polynomial weighting to positive vs. negative samples.
    """
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super(AsymmetricPolynomialLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # Calculate probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        # Asymmetric Clipping (ignores very easy negatives to prevent them from overwhelming the loss)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Polynomial weighting (Taylor expansion logic for robustness)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Apply asymmetric gammas
        loss = los_pos * (1 - xs_pos) ** self.gamma_pos + los_neg * (1 - xs_neg) ** self.gamma_neg
        
        return -loss.mean()