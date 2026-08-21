import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the FP16 operation order identical to the reference model.
        # This avoids compiler-fused SiLU rounding differences accumulating in
        # later decoder layers while preserving the production tensor shape.
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
