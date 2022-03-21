import torch
import torch.nn.functional as F


def make_criterion(alpha: float = 0.6, T: float = 4.0):
    def criterion(y_t, y_s, hard_targets):
        p_t = F.log_softmax(y_t / T, dim=1)  # teacher log softmax
        p_s = F.log_softmax(y_s / T, dim=1)  # student log softmax
        soft_loss = F.kl_div(p_s, p_t, reduction='sum') * (T ** 2) / y_s.shape[0]  # student teacher KV
        hard_loss = torch.nn.functional.cross_entropy(y_s, hard_targets)
        loss = alpha * soft_loss + (1. - alpha) * hard_loss
        return loss

    return criterion
