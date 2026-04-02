import torch
import numpy as np

# --------------------------------------------------
#  Mass coefficient matrix
# --------------------------------------------------

# Global cache for mass coefficient matrices to avoid recomputation
_mass_coeff_cache = {}

def build_mass_coeff_matrix(
    focal_sets,
    dtype=torch.bfloat16
):
    # Create a key based on focal sets structure to cache the matrix
    # Use id() for efficiency since focal_sets is typically static
    key = (id(focal_sets), str(dtype))

    if key in _mass_coeff_cache:
        return _mass_coeff_cache[key]

    mass_co = torch.zeros((len(focal_sets), len(focal_sets)), dtype=dtype)

    for i, A in enumerate(focal_sets):
        for j, B in enumerate(focal_sets):
            leng = 0
            if set(B).issubset(set(A)):
                leng = (-1) ** (len(A) - len(B))
            mass_co[j][i] = leng

    # Cache the matrix
    _mass_coeff_cache[key] = mass_co
    return mass_co

#Mobius inverse function
def belief_to_mass(
    mass,

):
    # Validate input for NaN/Inf values
    if torch.isnan(mass).any() or torch.isinf(mass).any():
        print(f"DEBUG: Found NaN/Inf in input mass to belief_to_mass")
        print(f"DEBUG: Input mass: {mass}")
        # Replace NaN/Inf with small positive values
        mass = torch.where(torch.isnan(mass) | torch.isinf(mass),
                          torch.tensor(1e-7, dtype=mass.dtype, device=mass.device),
                          mass)

    # Clamp negative values to avoid issues
    mass = torch.clamp(mass, min=1e-7)

    # Perform matrix multiplication using torch
    bel_mass = mass

    # Debug: Check for issues after matrix multiplication
    if torch.isnan(bel_mass).any() or torch.isinf(bel_mass).any():
        print(f"DEBUG: Found NaN/Inf in bel_mass after matrix multiplication")
        print(f"DEBUG: Bel mass after matmul: {bel_mass}")
        # Replace NaN/Inf with small positive values
        bel_mass = torch.where(torch.isnan(bel_mass) | torch.isinf(bel_mass),
                              torch.tensor(1e-7, dtype=bel_mass.dtype, device=bel_mass.device),
                              bel_mass)

    # Clamp negative values to 0
    bel_mass = torch.clamp(bel_mass, min=0.0)

    # Calculate sums for normalization
    sums_ = 1 - torch.sum(bel_mass, dim=-1)
    sums_ = torch.clamp(sums_, min=1e-7)  # Use a small positive value instead of 0 to avoid division by zero

    # Debug: Check sums_
    if torch.isnan(sums_).any() or torch.isinf(sums_).any():
        print(f"DEBUG: Found NaN/Inf in sums_ calculation")
        print(f"DEBUG: Bel mass before sums: {bel_mass}")
        print(f"DEBUG: Sums_: {sums_}")
        # Replace NaN/Inf with small positive values
        sums_ = torch.where(torch.isnan(sums_) | torch.isinf(sums_),
                           torch.tensor(1e-7, dtype=sums_.dtype, device=sums_.device),
                           sums_)

    # Append sums to the tensor
    sums_expanded = sums_.unsqueeze(-1)
    bel_mass = torch.cat([bel_mass, sums_expanded], dim=-1)

    # Normalize
    total_sum = torch.sum(bel_mass, dim=-1, keepdim=True)

    # Debug: Check total_sum for issues
    if torch.isnan(total_sum).any() or torch.isinf(total_sum).any():
        print(f"DEBUG: Found NaN/Inf in total_sum for normalization")
        print(f"DEBUG: Bel mass before normalization: {bel_mass}")
        print(f"DEBUG: Total sum: {total_sum}")
        # Replace NaN/Inf with small positive values
        total_sum = torch.where(torch.isnan(total_sum) | torch.isinf(total_sum),
                               torch.tensor(1e-7, dtype=total_sum.dtype, device=total_sum.device),
                               total_sum)

    # Avoid division by zero - use a more conservative minimum
    bel_mass = bel_mass / torch.clamp(total_sum, min=1e-7)

    # Debug: Final check
    if torch.isnan(bel_mass).any() or torch.isinf(bel_mass).any():
        print(f"DEBUG: Found NaN/Inf in final bel_mass")
        print(f"DEBUG: Final bel mass: {bel_mass}")
        # Replace NaN/Inf with small positive values
        bel_mass = torch.where(torch.isnan(bel_mass) | torch.isinf(bel_mass),
                              torch.tensor(1e-7, dtype=bel_mass.dtype, device=bel_mass.device),
                              bel_mass)

    # Final clamp to ensure all values are valid
    bel_mass = torch.clamp(bel_mass, min=1e-7, max=1.0)

    return bel_mass

def mass_to_pignistic(
    mass,
    focal_sets_full,
    vocab,
):
    # Validate input for NaN/Inf values
    if torch.isnan(mass).any() or torch.isinf(mass).any():
        print(f"DEBUG: Found NaN/Inf in mass values in mass_to_pignistic")
        print(f"DEBUG: Mass shape: {mass.shape}")
        print(f"DEBUG: Mass values: {mass}")
        # Replace NaN/Inf with small positive values
        mass = torch.where(torch.isnan(mass) | torch.isinf(mass),
                          torch.tensor(1e-7, dtype=mass.dtype, device=mass.device),
                          mass)

    # Clamp negative values to avoid issues
    mass = torch.clamp(mass, min=1e-7)

    # Create the betp matrix using torch - cache this as well
    betp_matrix_key = (len(focal_sets_full), len(vocab))

    # Initialize the cache if it doesn't exist
    if not hasattr(mass_to_pignistic, '_betp_matrix_cache'):
        mass_to_pignistic._betp_matrix_cache = {}
    if betp_matrix_key not in mass_to_pignistic._betp_matrix_cache:
        # Create the matrix for the first time
        betp_matrix = torch.zeros((len(focal_sets_full), len(vocab)), dtype=mass.dtype, device=mass.device)
        for i, c in enumerate(vocab):
            for j, A in enumerate(focal_sets_full):
                if set([c]).issubset(A):
                    betp_matrix[j, i] = 1.0 / len(A)

        # Store the original matrix
        mass_to_pignistic._betp_matrix_cache[betp_matrix_key] = {'original': betp_matrix.cpu()}

    # Get the original matrix
    original_matrix = mass_to_pignistic._betp_matrix_cache[betp_matrix_key]['original']

    # Device-specific caching
    device = mass.device
    dtype = mass.dtype

    # Create a cache key based on device and dtype
    device_cache_key = (str(device), str(dtype))

    # Check if we have the matrix cached for this device/dtype combination
    if 'device_cache' not in mass_to_pignistic._betp_matrix_cache[betp_matrix_key]:
        mass_to_pignistic._betp_matrix_cache[betp_matrix_key]['device_cache'] = {}

    if device_cache_key not in mass_to_pignistic._betp_matrix_cache[betp_matrix_key]['device_cache']:
        # Move the original matrix to the current device and dtype, then cache it
        betp_matrix = original_matrix.to(device).to(dtype)
        mass_to_pignistic._betp_matrix_cache[betp_matrix_key]['device_cache'][device_cache_key] = betp_matrix
    else:
        # Use the cached matrix
        betp_matrix = mass_to_pignistic._betp_matrix_cache[betp_matrix_key]['device_cache'][device_cache_key]

    # Debug: Check betp matrix for potential issues
    if torch.isnan(betp_matrix).any() or torch.isinf(betp_matrix).any():
        print(f"DEBUG: Found NaN/Inf in betp_matrix")
        print(f"DEBUG: Betp matrix: {betp_matrix}")
        # Replace NaN/Inf in betp_matrix
        betp_matrix = torch.where(torch.isnan(betp_matrix) | torch.isinf(betp_matrix),
                                 torch.tensor(1e-7, dtype=betp_matrix.dtype, device=betp_matrix.device),
                                 betp_matrix)

    # Perform matrix multiplication using torch
    final_bet_p = torch.matmul(mass, betp_matrix)

    # Validate final result for NaN/Inf values
    if torch.isnan(final_bet_p).any() or torch.isinf(final_bet_p).any():
        print(f"DEBUG: Found NaN/Inf in final betp")
        print(f"DEBUG: Mass values: {mass}")
        print(f"DEBUG: Betp matrix: {betp_matrix}")
        print(f"DEBUG: Final betp: {final_bet_p}")
        # Replace NaN/Inf with small positive values
        final_bet_p = torch.where(torch.isnan(final_bet_p) | torch.isinf(final_bet_p),
                                 torch.tensor(1e-7, dtype=final_bet_p.dtype, device=final_bet_p.device),
                                 final_bet_p)

    # Final clamp to ensure all values are valid
    final_bet_p = torch.clamp(final_bet_p, min=1e-7, max=1.0)

    return final_bet_p
