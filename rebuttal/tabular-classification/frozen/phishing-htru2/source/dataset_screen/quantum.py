"""Exact float64 IQP comparator features and chain-contracted fidelities.

Inputs are unscaled arctan angles. The scale acts before adjacent products.
For U=H D H, U**r=H D**r H. Both routines use this identity and never allocate
an exponential state. Output local features are qubit-major X,Y,Z triples.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
import torch


def _arguments(angles, scale, power):
    if not isinstance(scale, numbers.Real) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite scale required')
    if type(power) is not int or power not in (1, 2):
        raise ValueError('IQP power must be 1 or 2')
    if isinstance(angles, torch.Tensor):
        angles = angles.detach().cpu().numpy()
    raw = np.asarray(angles)
    if np.iscomplexobj(raw):
        raise ValueError('Real angles required')
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('Finite real angles[N,d], d>=1 required')
    with np.errstate(over='ignore', invalid='ignore'):
        x = x*scale
    if not np.isfinite(x).all():
        raise ValueError('Scaled angles must be finite')
    if x.shape[1] > 1 and not np.isfinite(x[:, :-1]*x[:, 1:]).all():
        raise ValueError('Pair phases must be finite')
    return x


def iqp_features(angles, scale, power):
    """NumPy float64 [N,3d] exact local X,Y,Z expectations of U**r|0>."""
    b = _arguments(angles, scale, power)
    factors = np.ones_like(b)
    edge = np.cos(power*b[:, :-1]*b[:, 1:])
    factors[:, :-1] *= edge
    factors[:, 1:] *= edge
    local = np.zeros((len(b), b.shape[1], 3), dtype=np.float64)
    local[:, :, 1] = -np.sin(power*b)*factors
    local[:, :, 2] = np.cos(power*b)*factors
    if not np.isfinite(local).all() or (abs(local) > 1+1e-12).any():
        raise ArithmeticError('Invalid local expectations')
    return local.reshape(len(b), 3*b.shape[1])


def _overlap(left, right, power):
    """Uniform spin-chain phase-difference average, complex128 [L,R]."""
    # <psi(left)|psi(right)>: left phase minus right phase.
    unary = left[:, None, 0]-right[None, :, 0]
    spins = left.new_tensor([1., -1.])
    transfer = .5*torch.exp(.5j*power*unary[:, :, None]*spins)
    pair_spins = spins[:, None]*spins[None, :]
    for j in range(1, left.shape[1]):
        coupling = left[:, None, j-1]*left[:, None, j]-right[None, :, j-1]*right[None, :, j]
        unary = left[:, None, j]-right[None, :, j]
        edge = torch.exp(.5j*power*coupling[:, :, None, None]*pair_spins)
        local = .5*torch.exp(.5j*power*unary[:, :, None]*spins)
        transfer = (transfer[:, :, :, None]*edge).sum(dim=-2)*local
    return transfer.sum(dim=-1)


@torch.no_grad()
def fidelity_kernel(left_angles, right_angles, scale, power, device='cpu', block_size=256):
    """NumPy float64 [N,M] fidelity with O(block_size**2) transfer memory."""
    left = _arguments(left_angles, scale, power)
    right = _arguments(right_angles, scale, power)
    if left.shape[1] != right.shape[1]:
        raise ValueError('Left/right feature counts must match')
    if type(block_size) is not int or block_size < 1:
        raise ValueError('Positive integer block_size required')
    device = torch.device(device)
    l = torch.as_tensor(left, dtype=torch.float64, device=device)
    r = torch.as_tensor(right, dtype=torch.float64, device=device)
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for a in range(0, len(left), block_size):
        for b in range(0, len(right), block_size):
            overlap = _overlap(l[a:a+block_size], r[b:b+block_size], power)
            value = overlap.abs().square()
            if not bool(torch.isfinite(value).all()) or bool((value > 1+1e-11).any()):
                raise ArithmeticError('Invalid normalized IQP overlap')
            result[a:a+block_size, b:b+block_size] = value.cpu().numpy()
    return result
