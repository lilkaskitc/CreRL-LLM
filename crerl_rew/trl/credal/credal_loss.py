import torch
import torch.nn.functional as F

class RSLLMLoss(torch.nn.Module):
    def __init__(self, alpha=1e-2, beta=1e-2):
        super().__init__()

        # Convert to float first in case they are strings (e.g., from command line args)
        self.alpha = torch.tensor([float(alpha)], dtype=torch.float32)  # Changed to float32 for better numerical stability
        self.beta = torch.tensor([float(beta)], dtype=torch.float32)    # Changed to float32 for better numerical stability

        # Cache for the mass coefficient matrix on different devices
        self._mass_coeff_matrix_cache = {}
        self._original_M = None

    def forward(self, belief_pred, belief_gt, M):
        # Store the original M if not already stored (first call)
        if self._original_M is None:
            self._original_M = M

        # Validate inputs for NaN/Inf values
        if torch.isnan(belief_pred).any() or torch.isinf(belief_pred).any():
            print(f"DEBUG: Found NaN/Inf in belief_pred in RSLLMLoss")
            # Replace NaN/Inf with small positive values
            belief_pred = torch.where(torch.isnan(belief_pred) | torch.isinf(belief_pred),
                                     torch.tensor(1e-7, dtype=belief_pred.dtype, device=belief_pred.device),
                                     belief_pred)

        # Clamp belief_pred to valid range with more conservative bounds
        belief_pred = torch.clamp(belief_pred, min=1e-7, max=1.0 - 1e-7)

        # Ensure belief_gt has the same dtype as belief_pred to avoid dtype mismatch
        belief_gt = belief_gt.to(belief_pred.dtype)

        # Validate belief_gt for NaN/Inf values
        if torch.isnan(belief_gt).any() or torch.isinf(belief_gt).any():
            print(f"DEBUG: Found NaN/Inf in belief_gt in RSLLMLoss")
            # Replace NaN/Inf with small positive values
            belief_gt = torch.where(torch.isnan(belief_gt) | torch.isinf(belief_gt),
                                   torch.tensor(1e-7, dtype=belief_gt.dtype, device=belief_gt.device),
                                   belief_gt)

        # Clamp belief_gt to valid range with more conservative bounds
        belief_gt = torch.clamp(belief_gt, min=1e-7, max=1.0 - 1e-7)

        # (8) BCE on belief - add epsilon for numerical stability
        eps = 1e-7
        clamped_belief_pred = torch.clamp(belief_pred, min=eps, max=1.0 - eps)
        clamped_belief_gt = torch.clamp(belief_gt, min=eps, max=1.0 - eps)

        bce = F.binary_cross_entropy(
            clamped_belief_pred, clamped_belief_gt, reduction="mean"
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

        # Validate mass for NaN/Inf values
        if torch.isnan(mass).any() or torch.isinf(mass).any():
            print(f"DEBUG: Found NaN/Inf in mass after matrix multiplication in RSLLMLoss")
            # Replace NaN/Inf with small positive values
            mass = torch.where(torch.isnan(mass) | torch.isinf(mass),
                             torch.tensor(1e-7, dtype=mass.dtype, device=mass.device),
                             mass)

        # Clamp mass values to avoid extreme values
        mass = torch.clamp(mass, min=-100.0, max=100.0)

        alpha = self.alpha.to(belief_pred.device)
        beta = self.beta.to(belief_pred.device)

        # (9) Mass regularization - use more numerically stable approach
        mr = torch.relu(-mass).mean()

        # (10) Mass sum constraint - use more numerically stable approach
        mass_sum = mass.sum(dim=-1)
        ms = torch.relu(mass_sum.mean() - 1.0)

        return bce + alpha * mr + beta * ms, mass
