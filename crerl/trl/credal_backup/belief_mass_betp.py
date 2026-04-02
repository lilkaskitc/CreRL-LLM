import torch
import numpy as np

# --------------------------------------------------
# 4. Mass coefficient matrix
# --------------------------------------------------

def build_mass_coeff_matrix(
    focal_sets,
    dtype=torch.bfloat16
):
    mass_co = torch.zeros((len(focal_sets), len(focal_sets)), dtype=dtype)

    for i, A in enumerate(focal_sets):
        for j, B in enumerate(focal_sets):
            leng = 0
            if set(B).issubset(set(A)):
                leng = (-1) ** (len(A) - len(B))
            mass_co[j][i] = leng
    return mass_co

#Mobius inverse function
def belief_to_mass(
    bel,
    focal_sets,
):
    mass_coeff_matrix = build_mass_coeff_matrix(focal_sets, dtype=bel.dtype).to(bel.device)

    # Perform matrix multiplication using torch
    bel_mass = torch.matmul(bel, mass_coeff_matrix)

    # Clamp negative values to 0
    bel_mass = torch.clamp(bel_mass, min=0.0)

    # Calculate sums for normalization
    sums_ = 1 - torch.sum(bel_mass, dim=-1)
    sums_ = torch.clamp(sums_, min=0.0)

    # Append sums to the tensor
    sums_expanded = sums_.unsqueeze(-1)
    bel_mass = torch.cat([bel_mass, sums_expanded], dim=-1)

    # Normalize
    total_sum = torch.sum(bel_mass, dim=-1, keepdim=True)
    bel_mass = bel_mass / total_sum

    return bel_mass

def mass_to_pignistic(
    mass,
    focal_sets_full,
    vocab,
):
    # Create the betp matrix using torch
    betp_matrix = torch.zeros((len(focal_sets_full), len(vocab)), dtype=mass.dtype, device=mass.device)
    for i,c in enumerate(vocab):
        for j,A in enumerate(focal_sets_full):
            if set([c]).issubset(A):
                betp_matrix[j,i] = 1/len(A)

    # Perform matrix multiplication using torch
    final_bet_p = torch.matmul(mass, betp_matrix)

    return final_bet_p
