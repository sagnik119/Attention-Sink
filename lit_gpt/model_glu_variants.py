"""
GLU Activation Function Variants for FFN layers.

This module implements various GLU (Gated Linear Unit) variants to replace SwiGLU
in the FFN layers while keeping softmax attention unchanged.

Variants implemented:
1. SiGLU (bounded GLU) - sigmoid gating with bounded output
2. TanhGLU (signed, bounded) - tanh gating allowing sign while bounded
3. Capped-SwiGLU - SwiGLU with smooth capping
4. Norm-GLU - token-wise gate normalization
5. Additive-gate variant - additive rescaling instead of multiplicative
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lit_gpt.config import Config


class SiGLUMLP(nn.Module):
    """
    SiGLU (bounded GLU): h = (W_up * x) ⊙ σ(β * W_gate * x)
    
    The sigmoid gate ∈ (0,1) ensures ||h|| scales at most linearly with ||x||.
    β is tuned to match SwiGLU's gate variance at initialization.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.fc_up = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_gate = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        
        # β parameter to match SwiGLU's gate variance at initialization
        # SwiGLU uses SiLU(x) which has variance ≈ 0.596 for standard normal input
        # Sigmoid has variance = 1/12 ≈ 0.083 for uniform input, so we need β ≈ sqrt(0.596/0.083) ≈ 2.68
        self.beta = nn.Parameter(torch.tensor(2.68))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_proj = self.fc_up(x)
        gate_proj = self.fc_gate(x)
        # Apply sigmoid with β scaling to match SwiGLU variance
        gate_activated = torch.sigmoid(self.beta * gate_proj)
        # Element-wise multiplication (Hadamard product)
        gated = up_proj * gate_activated
        return self.proj(gated)


class TanhGLUMLP(nn.Module):
    """
    TanhGLU (signed, bounded): h = (W_up * x) ⊙ tanh(β * W_gate * x)
    
    Allows sign while still bounded; often slightly worse perplexity than SwiGLU
    but good for demonstrating bounded activation effects.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.fc_up = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_gate = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        
        # β parameter to match SwiGLU's gate variance at initialization
        # Tanh has variance ≈ 1 - (2/π) ≈ 0.363 for standard normal input
        # We want to match SwiGLU's variance ≈ 0.596, so β ≈ sqrt(0.596/0.363) ≈ 1.28
        self.beta = nn.Parameter(torch.tensor(1.28))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_proj = self.fc_up(x)
        gate_proj = self.fc_gate(x)
        # Apply tanh with β scaling
        gate_activated = torch.tanh(self.beta * gate_proj)
        # Element-wise multiplication
        gated = up_proj * gate_activated
        return self.proj(gated)


class CappedSwiGLUMLP(nn.Module):
    """
    Capped-SwiGLU: h = (W_up * x) ⊙ min(SiLU(W_gate * x), c)
    
    Uses a smooth cap based on softplus to limit the maximum activation value.
    The cap c is learned during training.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.fc_up = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_gate = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        
        # Learnable cap parameter, initialized to a reasonable value
        # Start with cap = 6.0 (similar to ReLU6 but softer)
        self.cap = nn.Parameter(torch.tensor(6.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_proj = self.fc_up(x)
        gate_proj = self.fc_gate(x)
        
        # Apply SiLU activation
        silu_activated = F.silu(gate_proj)
        
        # Apply smooth capping using softplus-based smooth minimum
        # smooth_min(a, b) ≈ -softplus(-a - b) + a + b for a ≈ b
        # For numerical stability, we use: min(silu, cap) ≈ silu - softplus(silu - cap)
        capped_gate = silu_activated - F.softplus(silu_activated - self.cap)
        
        # Element-wise multiplication
        gated = up_proj * capped_gate
        return self.proj(gated)


class NormGLUMLP(nn.Module):
    """
    Norm-GLU (token-wise gate normalization): Preserves expressivity but removes 
    magnitude degrees of freedom that drive tails.
    
    h = (W_up * x) ⊙ normalize(SiLU(W_gate * x))
    
    The gate activations are normalized per token to have unit norm, removing
    the magnitude scaling while preserving directional information.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.fc_up = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_gate = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        
        # Small epsilon for numerical stability in normalization
        self.eps = 1e-8
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_proj = self.fc_up(x)
        gate_proj = self.fc_gate(x)
        
        # Apply SiLU activation
        silu_activated = F.silu(gate_proj)
        
        # Token-wise normalization: normalize along the intermediate_size dimension
        # Shape: [batch_size, seq_len, intermediate_size]
        gate_norm = torch.norm(silu_activated, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        normalized_gate = silu_activated / (gate_norm + self.eps)
        
        # Element-wise multiplication
        gated = up_proj * normalized_gate
        return self.proj(gated)


class AdditiveGateMLP(nn.Module):
    """
    Additive-gate variant (no product): h = (1 + α * tanh(β * W_gate * x)) ⊙ (W_up * x)
    
    Behaves like a bounded rescaling rather than a product of two linears,
    eliminating quadratic scaling. Uses small α for subtle modulation.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.fc_up = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_gate = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        
        # Small α for subtle modulation (typically 0.1 to 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.2))
        
        # β parameter for gate scaling
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_proj = self.fc_up(x)
        gate_proj = self.fc_gate(x)
        
        # Additive gating: (1 + α * tanh(β * gate))
        gate_modulation = 1.0 + self.alpha * torch.tanh(self.beta * gate_proj)
        
        # Element-wise multiplication with additive gate
        gated = gate_modulation * up_proj
        return self.proj(gated)


# Additional utility function for smooth capping (alternative implementation)
def smooth_cap(x: torch.Tensor, cap: float, smoothness: float = 1.0) -> torch.Tensor:
    """
    Smooth capping function using a differentiable approximation to min(x, cap).
    
    Args:
        x: Input tensor
        cap: Maximum value
        smoothness: Controls the smoothness of the transition (higher = smoother)
    
    Returns:
        Smoothly capped tensor
    """
    return cap * torch.tanh(x / cap * smoothness)


class SmoothCappedSwiGLUMLP(nn.Module):
    """
    Alternative implementation of Capped-SwiGLU using the smooth_cap function.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.fc_up = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_gate = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        
        # Learnable cap and smoothness parameters
        self.cap = nn.Parameter(torch.tensor(6.0))
        self.smoothness = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_proj = self.fc_up(x)
        gate_proj = self.fc_gate(x)
        
        # Apply SiLU activation then smooth capping
        silu_activated = F.silu(gate_proj)
        capped_gate = smooth_cap(silu_activated, self.cap, self.smoothness)
        
        # Element-wise multiplication
        gated = up_proj * capped_gate
        return self.proj(gated)