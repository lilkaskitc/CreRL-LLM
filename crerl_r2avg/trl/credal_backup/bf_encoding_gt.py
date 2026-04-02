import torch

def encode_gt_belief(
    gt_token,
    focal_sets,
    device,
):
    """
    Encodes ground truth as belief vector over focal sets.

    gt_token: correct token index
    """

    K = len(focal_sets)
    bel_gt = torch.zeros(K, device=device, dtype=torch.int)

    for i, A in enumerate(focal_sets):
        if gt_token in A:
            bel_gt[i] = 1
            
    return bel_gt
