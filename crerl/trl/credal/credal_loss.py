import torch
import torch.nn.functional as F

class RSLLMLoss(torch.nn.Module):
    def __init__(self, alpha=1e-2, beta=1e-2):
        super().__init__()

        # Convert to float first in case they are strings (e.g., from command line args)
        self.alpha = torch.tensor([float(alpha)], dtype=torch.float16)
        self.beta = torch.tensor([float(beta)], dtype=torch.float16)

        # Cache for the mass coefficient matrix on different devices
        self._mass_coeff_matrix_cache = {}
        self._original_M = None

    def forward(self, belief_pred, belief_gt, M):
        # Store the original M if not already stored (first call)
        if self._original_M is None:
            self._original_M = M

        # Ensure belief_gt has the same dtype as belief_pred to avoid dtype mismatch
        belief_gt = belief_gt.to(belief_pred.dtype)

        # (8) BCE on belief
        bce = F.binary_cross_entropy(
            belief_pred, belief_gt, reduction="mean"
        )

        # Get the mass coefficient matrix for the current device and dtype
        device = belief_pred.device
        dtype = belief_pred.dtype

        # Create a cache key based on device and dtype
        cache_key = (str(device), str(dtype))

        # Check if we have the matrix cached for this device/dtype combination
        if cache_key not in self._mass_coeff_matrix_cache:
            # Move the original matrix to the current device and dtype, then cache it
            mass_coeff_matrix = self._original_M.to(device).to(dtype)
            self._mass_coeff_matrix_cache[cache_key] = mass_coeff_matrix
        else:
            # Use the cached matrix
            mass_coeff_matrix = self._mass_coeff_matrix_cache[cache_key]

        # Use the cached matrix (already on correct device and dtype)
        mass = torch.matmul(belief_pred, mass_coeff_matrix)
        alpha = self.alpha.to(belief_pred.device)
        beta = self.beta.to(belief_pred.device)

        # (9) Mass regularization
        mr = torch.relu(-mass).mean()

        # (10) Mass sum constraint
        ms = torch.relu(mass.sum(dim=-1).mean() - 1.0)

        return bce + alpha * mr + beta * ms, mass
