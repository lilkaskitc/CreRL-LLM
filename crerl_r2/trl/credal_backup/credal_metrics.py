import torch
import numpy as np
from  .belief_mass_betp  import mass_to_pignistic

def compute_credal_metrics(
    belief_mass,
    vocab,
    focal_sets_full,
):
    # 1. Pignistic entropy
    betp = mass_to_pignistic(belief_mass, focal_sets_full, vocab)

    # Avoid log(0) by setting it to a very small value
    eps = 1e-8
    pignistic_entropy = -torch.sum(betp * torch.log2(torch.clamp(betp, min=eps, max=1.0)), dim=-1)

    # 2. Credal width
    # Convert tensor operations to numpy for compute_vertices function
    class_indices = torch.argmax(betp, dim=-1).cpu().numpy()
    vertices_np = compute_vertices(belief_mass, vocab, class_indices, focal_sets_full)

    # Convert back to tensor for final operations
    vertices = torch.tensor(vertices_np, dtype=belief_mass.dtype, device=belief_mass.device)
    credal_width = torch.max(vertices, dim=-1)[0] - torch.min(vertices, dim=-1)[0]

    return {
        "pignistic_entropy": pignistic_entropy,
        "credal_width": credal_width,
    }


# Compute vertices of the credal set
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