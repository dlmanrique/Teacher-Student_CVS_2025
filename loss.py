import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, class_weights, T=2.0, alpha=0.5, device="cuda"):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.criterion_gt = nn.BCEWithLogitsLoss(weight=class_weights.to(device))

    def forward(self, student_logits, teacher_logits, targets):
        # Hard loss con ground truth
        hard_loss = self.criterion_gt(student_logits, targets.float())

        # Soft loss con distillation
        # -> aplicar sigmoid con temperatura
        teacher_probs = torch.sigmoid(teacher_logits / self.T)
        student_probs = torch.sigmoid(student_logits / self.T)

        # puedes usar KL o MSE, aquí te dejo MSE (más estable en multilabel)
        soft_loss = F.mse_loss(student_probs, teacher_probs)

        # combinar
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss