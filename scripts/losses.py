# losses.py

import torch
import torch.nn.functional as F

POS_WEIGHT = 2.0


def binary_loss_fn(logits, targets):
    pos_weight = torch.tensor(
        [POS_WEIGHT], device=logits.device
    )
    return F.binary_cross_entropy_with_logits(
        logits.view(-1),
        targets.float(),
        pos_weight=pos_weight
    )


def listnet_loss(pred_scores, true_labels):
    """
    ListNet loss computed over a batch.
    Assumes each batch represents a ranking list.
    """
    pred_scores = pred_scores.view(-1)
    true_labels = true_labels.view(-1)

    P_pred = F.softmax(pred_scores, dim=0)
    P_true = F.softmax(true_labels.float(), dim=0)

    return -(P_true * torch.log(P_pred + 1e-8)).sum()
