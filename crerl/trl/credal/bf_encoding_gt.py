import torch

def encode_gt_belief(
    gt_token,
    focal_sets,
    device,
    transformation_matrix=None,
):
    """
    Encodes ground truth as belief vector over focal sets.

    gt_token: correct token index
    """
    # Convert gt_token to tensor if it's not already
    if not torch.is_tensor(gt_token):
        gt_token = torch.tensor(gt_token, device=device)

    # Use cached transformation matrix if available for efficient lookup
    if transformation_matrix is not None:
        # transformation_matrix should have shape [vocab_size, n_focals]
        # where each entry [token_idx, focal_idx] is True if token belongs to that focal set
        if gt_token < transformation_matrix.size(0):
            # Get the row for the ground truth token and convert to int
            bel_gt = transformation_matrix[gt_token].int()
            return bel_gt.to(device)
        else:
            # If token index is out of bounds, fall back to original method
            pass

    K = len(focal_sets)

    # Create a tensor to store the result
    bel_gt = torch.zeros(K, device=device, dtype=torch.int)

    # Check membership using set operations (more efficient for lookups)
    for i, focal_set in enumerate(focal_sets):
        if gt_token in focal_set:
            bel_gt[i] = 1

    return bel_gt
