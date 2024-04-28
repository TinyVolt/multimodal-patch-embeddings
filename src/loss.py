import torch
import torch.nn.functional as F

def calculate_normalized_l1_loss(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
    norm: float = 1
) -> torch.Tensor:
    '''
    Instead of `loss(y_pred_normalized.mul(norm), y_target_normalized.mul(norm))`,
    we multiply the final loss with `norm` because it does not change the value and involves fewer flops
    '''
    y_pred_normalized = y_pred / y_pred.norm(dim=-1, keepdim=True)
    y_target_normalized = y_target / y_target.norm(dim=-1, keepdim=True)
    return F.l1_loss(y_pred_normalized, y_target_normalized, reduction='mean') * norm