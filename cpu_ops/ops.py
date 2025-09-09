from __future__ import annotations
from math import pi

import numpy as np
from typing import Tuple

def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm in fp32.

    Implement here. Expected shapes: x [..., D], gamma [D]. Return same shape as x.
    """
    
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    
    # Normalize and scale by gamma
    normalized = x / rms
    return normalized * gamma

def linear(x: np.ndarray, W: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """Naive linear: y = x @ W.T (+ bias).

    Shapes: x [N, D_in], W [D_out, D_in], bias [D_out]. Return [N, D_out].
    """
    
    output = x @ W.T
    
    
    if bias is not None:
        output = output + bias
        
    return output

    


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    
    sigmoid = 1 / (1 + np.exp(-x))

    x = x*sigmoid

    return x
    


def gelu(x: np.ndarray, *, approximate: bool = True) -> np.ndarray:
    """GELU activation. If approximate, use tanh approximation."""
    
    term = 1 + np.tanh(np.sqrt(2/pi)*(x + 0.044715*(x**3)))

    return 0.5*x*term 



def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax along given axis (rowwise when axis=-1 for 2D)."""
    
    exp_vec = np.exp(-x)

    x = x / exp_vec 

    return x


def rope_apply(q: np.ndarray, k: np.ndarray, pos: np.ndarray, *, rotary_dim: int | None = None, theta_base: float = 10000.0):
    # pos should be position indices [seq_len] or similar
    # Compute theta values and cos/sin internally
    if rotary_dim is None:
        rotary_dim = q.shape[-1]
    
    # Generate theta for each dimension pair
    dim_pairs = rotary_dim // 2
    theta = theta_base ** (-2 * np.arange(dim_pairs) / rotary_dim)
    
    # Compute angles: pos[:, None] * theta[None, :]
    angles = pos[..., None] * theta
    cos_vals = np.cos(angles)[..., None]  # Add pair dimension
    sin_vals = np.sin(angles)[..., None]
    
    # Rest of your rotation logic...
    # Extract the portion to apply rotation to
    q_rot = q[..., :rotary_dim]
    k_rot = k[..., :rotary_dim]
    
    # Split into pairs for rotation
    q_pairs = q_rot.reshape(*q_rot.shape[:-1], rotary_dim // 2, 2)
    k_pairs = k_rot.reshape(*k_rot.shape[:-1], rotary_dim // 2, 2)
    
    # Apply rotation: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
    q_x, q_y = q_pairs[..., 0:1], q_pairs[..., 1:2]
    k_x, k_y = k_pairs[..., 0:1], k_pairs[..., 1:2]
    
    q_rot_x = q_x * cos_vals - q_y * sin_vals
    q_rot_y = q_x * sin_vals + q_y * cos_vals
    k_rot_x = k_x * cos_vals - k_y * sin_vals
    k_rot_y = k_x * sin_vals + k_y * cos_vals
    
    # Recombine rotated pairs
    q_rotated = np.concatenate([q_rot_x, q_rot_y], axis=-1)
    k_rotated = np.concatenate([k_rot_x, k_rot_y], axis=-1)
    
    # Reshape back to original dimensions
    q_rotated = q_rotated.reshape(*q.shape[:-1], rotary_dim)
    k_rotated = k_rotated.reshape(*k.shape[:-1], rotary_dim)
    
    # Combine with non-rotated dimensions if any
    if rotary_dim < q.shape[-1]:
        q_out = np.concatenate([q_rotated, q[..., rotary_dim:]], axis=-1)
        k_out = np.concatenate([k_rotated, k[..., rotary_dim:]], axis=-1)
    else:
        q_out = q_rotated
        k_out = k_rotated
    
    return q_out, k_out
    
    




