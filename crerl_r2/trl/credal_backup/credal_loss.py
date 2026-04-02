import torch
import torch.nn.functional as F
from .belief_mass_betp import build_mass_coeff_matrix

class RSLLMLoss(torch.nn.Module):
    def __init__(self, alpha=1e-2, beta=1e-2):
        super().__init__()

        # Convert to float first in case they are strings (e.g., from command line args)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, belief_pred, belief_gt, focal_sets):
        # Ensure belief_gt has the same dtype as belief_pred to avoid dtype mismatch
        belief_gt = belief_gt.to(belief_pred.dtype)

        # (8) BCE on belief
        bce = F.binary_cross_entropy(
            belief_pred, belief_gt, reduction="mean"
        )

        # mass
        mass_coeff_matrix = build_mass_coeff_matrix(focal_sets, dtype=belief_pred.dtype).to(belief_pred.device)
        mass = torch.matmul(belief_pred, mass_coeff_matrix)

        # (9) Mass regularization
        mr = torch.relu(-mass).mean()

        # (10) Mass sum constraint
        ms = torch.relu(mass.sum(dim=-1).mean() - 1.0)

        return bce + self.alpha * mr + self.beta * ms
