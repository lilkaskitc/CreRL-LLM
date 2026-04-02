import torch
import numpy as np
from .belief_mass_betp import mass_to_pignistic

# Global cache for mask rows to avoid recomputation
_mask_row_cache = {}

def compute_credal_metrics(
    belief_mass,
    vocab,
    focal_sets_full,
):
    # Validate input for NaN/Inf values
    if torch.isnan(belief_mass).any() or torch.isinf(belief_mass).any():
        print(f"DEBUG: Found NaN/Inf in belief_mass")
        print(f"DEBUG: Belief mass shape: {belief_mass.shape}")
        print(f"DEBUG: Belief mass: {belief_mass}")
        # Replace NaN/Inf with small positive values
        belief_mass = torch.where(torch.isnan(belief_mass) | torch.isinf(belief_mass),
                                 torch.tensor(1e-7, dtype=belief_mass.dtype, device=belief_mass.device),
                                 belief_mass)

    # Clamp negative values to avoid issues
    belief_mass = torch.clamp(belief_mass, min=1e-7)


    # Determine if we have a batch dimension or single sample
    has_batch_dim = belief_mass.dim() > 1 or (belief_mass.dim() == 1 and len(focal_sets_full) != belief_mass.size(-1))

    if not has_batch_dim:
        # Single sample case: belief_mass has shape (num_focals,)
        # Add a batch dimension temporarily for consistent processing
        belief_mass = belief_mass.unsqueeze(0)  # Shape becomes (1, num_focals)

    # 1. Pignistic entropy
    betp = mass_to_pignistic(belief_mass, focal_sets_full, vocab)

    # Validate betp for NaN/Inf values
    if torch.isnan(betp).any() or torch.isinf(betp).any():
        print(f"DEBUG: Found NaN/Inf in betp from mass_to_pignistic")
        print(f"DEBUG: Betp: {betp}")
        # Replace NaN/Inf with small positive values
        betp = torch.where(torch.isnan(betp) | torch.isinf(betp),
                          torch.tensor(1e-7, dtype=betp.dtype, device=betp.device),
                          betp)

    # Avoid log(0) by setting it to a very small value
    eps = 1e-7
    clamped_betp = torch.clamp(betp, min=eps, max=1.0)
    log_betp = torch.log2(clamped_betp)

    # Debug: Check for issues in intermediate calculations
    if torch.isnan(log_betp).any() or torch.isinf(log_betp).any():
        print(f"DEBUG: Found NaN/Inf in log_betp calculation")
        print(f"DEBUG: Original betp: {betp}")
        print(f"DEBUG: Clamped betp: {clamped_betp}")
        print(f"DEBUG: Log betp: {log_betp}")
        # Replace NaN/Inf in log_betp
        log_betp = torch.where(torch.isnan(log_betp) | torch.isinf(log_betp),
                              torch.tensor(-23.0, dtype=log_betp.dtype, device=log_betp.device),  # log2(1e-7) ~ -23
                              log_betp)

    pignistic_entropy = -torch.sum(betp * log_betp, dim=-1)

    # Find the most likely token according to betp (argmax(betp))
    most_likely_token_idx = torch.argmax(betp, dim=-1)  # Shape: (batch_size,)

    # 2. Credal width - efficient computation using cached mask rows for the most likely token
    batch_size = belief_mass.size(0)
    credal_width_list = []

    # Create mapping from token_id to index in vocab once
    token_to_idx = {token_id: idx for idx, token_id in enumerate(vocab)}

    for i in range(batch_size):
        # Get the most likely token index for this sample
        # Handle the case where most_likely_token_idx is 0-dimensional
        if most_likely_token_idx.dim() == 0:
            # If most_likely_token_idx is a scalar tensor
            token_idx = most_likely_token_idx.item()
        else:
            # If most_likely_token_idx is a 1+ dimensional tensor
            token_idx = most_likely_token_idx[i].item()

        # Create a cache key for this specific token and device/dtype combination
        device = belief_mass.device
        dtype = belief_mass.dtype
        cache_key = (id(focal_sets_full), id(vocab), token_idx, str(device), str(dtype))

        # Check if we have the mask rows cached for this token
        if cache_key not in _mask_row_cache:
            # Compute mask rows only for the specific token index
            singleton_mask_for_token, contains_mask_for_token = cw_mask_single(token_idx, focal_sets_full, token_to_idx)

            # Move to the correct device and dtype, then cache them
            singleton_mask_for_token = singleton_mask_for_token.to(device).to(dtype)
            contains_mask_for_token = contains_mask_for_token.to(device).to(dtype)

            _mask_row_cache[cache_key] = (singleton_mask_for_token, contains_mask_for_token)
        else:
            # Use the cached mask rows
            singleton_mask_for_token, contains_mask_for_token = _mask_row_cache[cache_key]

        # Compute lower and upper bounds for this specific token
        # belief_mass[i] has shape (num_focals,), masks have shape (num_focals,)
        lower_val = torch.sum(belief_mass[i] * singleton_mask_for_token)
        upper_val = torch.sum(belief_mass[i] * contains_mask_for_token)

        # Compute credal width for this token
        width = upper_val - lower_val
        credal_width_list.append(width)

    # Stack the results to get the final credal_width tensor
    credal_width = torch.stack(credal_width_list)


    # Validate final results for NaN/Inf values
    if torch.isnan(pignistic_entropy).any() or torch.isinf(pignistic_entropy).any():
        print(f"DEBUG: Found NaN/Inf in final pignistic_entropy")
        print(f"DEBUG: Pignistic entropy: {pignistic_entropy}")
        print(f"DEBUG: Betp: {betp}")
        # Replace NaN/Inf with small positive values
        pignistic_entropy = torch.where(torch.isnan(pignistic_entropy) | torch.isinf(pignistic_entropy),
                                       torch.tensor(0.0, dtype=pignistic_entropy.dtype, device=pignistic_entropy.device),
                                       pignistic_entropy)

    if torch.isnan(credal_width).any() or torch.isinf(credal_width).any():
        print(f"DEBUG: Found NaN/Inf in final credal_width")
        print(f"DEBUG: Credal width: {credal_width}")
        print(f"DEBUG: Lower: {lower}, Upper: {upper}")
        # Replace NaN/Inf with small positive values
        credal_width = torch.where(torch.isnan(credal_width) | torch.isinf(credal_width),
                                  torch.tensor(0.0, dtype=credal_width.dtype, device=credal_width.device),
                                  credal_width)

    # Final clamping to ensure reasonable values
    pignistic_entropy = torch.clamp(pignistic_entropy, min=0.0)
    credal_width = torch.clamp(credal_width, min=0.0)

    # If we had a single sample originally, remove the batch dimension from results
    if not has_batch_dim:
        pignistic_entropy = pignistic_entropy.squeeze(0)  # Remove batch dimension
        credal_width = credal_width.squeeze(0)  # Remove batch dimension

    return {
        "pignistic_entropy": pignistic_entropy,
        "credal_width": credal_width,
    }

def cw_mask_single(target_token_idx, focal_sets_full, token_to_idx):
    """
    Create mask rows for a single token for efficient credal width computation.

    Args:
        target_token_idx: Index of the target token in the vocab
        focal_sets_full: List of focal sets (each focal set is a set of token IDs)
        token_to_idx: Dictionary mapping token_id to index in vocab

    Returns:
        mask_singleton: Vector of shape (len(focal_sets_full),)
                        mask_singleton[j] = 1 if focal j == {target token}, 0 otherwise
        mask_contains: Vector of shape (len(focal_sets_full),)
                       mask_contains[j] = 1 if target token ∈ focal j, 0 otherwise
    """
    num_focals = len(focal_sets_full)

    # Find the token_id corresponding to the target index
    target_token_id = None
    for token_id, idx in token_to_idx.items():
        if idx == target_token_idx:
            target_token_id = token_id
            break

    if target_token_id is None:
        raise ValueError(f"Target token index {target_token_idx} not found in vocab")

    # Initialize mask vectors
    mask_singleton = torch.zeros((num_focals,), dtype=torch.float16)
    mask_contains = torch.zeros((num_focals,), dtype=torch.float16)

    # Fill the mask vectors for the target token only
    for j, focal_set in enumerate(focal_sets_full):
        if target_token_id in focal_set:
            mask_contains[j] = 1.0
            # Check if this focal set is a singleton containing only the target token
            if len(focal_set) == 1:
                mask_singleton[j] = 1.0

    return mask_singleton, mask_contains

def cw_mask(vocab, focal_sets_full):
    """
    Create mask matrices for efficient credal width computation.

    Args:
        vocab: List of token IDs
        focal_sets_full: List of focal sets (each focal set is a set of token IDs)

    Returns:
        mask_singleton: Matrix of shape (len(focal_sets_full), len(vocab))
                        mask_singleton[j, i] = 1 if focal j == {token i}, 0 otherwise
        mask_contains: Matrix of shape (len(focal_sets_full), len(vocab))
                       mask_contains[j, i] = 1 if token i ∈ focal j, 0 otherwise
    """
    num_focals = len(focal_sets_full)
    vocab_size = len(vocab)

    # Create mapping from token_id to index in vocab
    token_to_idx = {token_id: idx for idx, token_id in enumerate(vocab)}

    # Initialize mask matrices
    mask_singleton = torch.zeros((num_focals, vocab_size), dtype=torch.float16)
    mask_contains = torch.zeros((num_focals, vocab_size), dtype=torch.float16)

    # Fill the mask matrices
    for j, focal_set in enumerate(focal_sets_full):
        for token_id in focal_set:
            if token_id in token_to_idx:
                token_idx = token_to_idx[token_id]
                mask_contains[j, token_idx] = 1.0
                # Check if this focal set is a singleton
                if len(focal_set) == 1:
                    mask_singleton[j, token_idx] = 1.0

    return mask_singleton, mask_contains

# Simplified version that's more efficient - not currently used but kept for reference
def compute_vertices(mass, classes, class_index, new_classes_with_full):
    vertices = []
    num_permutations = 0
    subset_permutations = []
    class_indexes = np.arange(len(classes))
    classes_array = np.array(classes)
    for _ in range(num_permutations):
      np.random.shuffle(class_indexes)
      subset_permutations.append(list(classes_array[class_indexes]))

    p_0 = classes.copy()
    p_0[0], p_0[class_index] = p_0[class_index], p_0[0]

    p_1 = classes.copy()
    p_1[-1], p_1[class_index] = p_1[class_index], p_1[-1]

    subset_permutations = [p_0] + [p_1] + subset_permutations

    for perm in subset_permutations:
        p = 0
        i, curr_elem = perm.index(classes[class_index]), classes[class_index]
        for c in new_classes_with_full:
          if curr_elem in c:
            for k in range(i):
              if perm[k] in c:
                break
            else:
              # Handle both tensor and numpy array mass
              if torch.is_tensor(mass):
                  p += mass[new_classes_with_full.index(c)].item()
              else:
                  p += mass[new_classes_with_full.index(c)]
        vertices.append(p)

    return np.array(vertices)