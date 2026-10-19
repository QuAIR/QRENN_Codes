"""Exact product-probe moments for the MAGIC encoding comparison.

Inputs are the unscaled transformed features. Scale multiplies all features
before encoding, including before IQP pair products. Angle uses tensor Rx;
Diagonal uses positive phases, pads phases with zeros, and retains the global
phase seen by the controlled oracle. Diagonal needs ceil(log2(d)) qubits
(at least one); the other encoders use d. Qubit zero is the most significant.

The immutable root model supplies IQP/Pauli moments and the m=1 processing
contraction: exp(+i theta.Pauli), zero processing input, branch-one controlled
queries alternating U/U-dagger, and final processing-Z score favoring label1.
No dense data-register oracle is constructed. Autograd retains x and probe
dependence so final fits can continue joint optimization of the calibrated
probe and processing parameters.
"""
from __future__ import annotations

import math
import numbers

import torch

from model import moments as _root_moments
from model import scores_from_moments

_ENCODINGS = ('angle', 'diagonal', 'iqp', 'pauli')


def data_qubits(features: int, encoding: str) -> int:
    """Return the register size without truncating any input features."""
    if encoding not in _ENCODINGS:
        raise ValueError('encoding must be angle, diagonal, iqp or pauli.')
    if not isinstance(features, int) or isinstance(features, bool) or features < 1:
        raise ValueError('features must be a positive integer.')
    return max(1, (features-1).bit_length()) if encoding == 'diagonal' else features


def _real_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f'{name} must be a float32 or float64 tensor.')


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f'{name} must contain finite values.')


def prepare_probe(probe: torch.Tensor) -> torch.Tensor:
    """Return tensor_j Ry(probe[j])|0>, normalized and in big-endian order."""
    _real_tensor(probe, 'probe')
    if probe.ndim != 1 or probe.numel() < 1:
        raise ValueError('Expected probe[n], n>=1.')
    _finite(probe, 'probe')
    cdtype = torch.complex128 if probe.dtype == torch.float64 else torch.complex64
    state = torch.ones(1, dtype=cdtype, device=probe.device)
    for angle in probe:
        state = torch.kron(state, torch.stack((torch.cos(angle/2), torch.sin(angle/2))).to(cdtype))
    return state


def moments(x: torch.Tensor, probe: torch.Tensor, encoding: str,
            max_order: int, scale: float = 1.) -> torch.Tensor:
    """Return mu[:,k]=<probe|U(scale*x)**k|probe> for k=0,...,K.

    Negative orders are conjugates of positive orders. The output is [N,K+1]
    complex64/128 according to the promoted real input/probe dtype.
    """
    _real_tensor(x, 'x')
    _real_tensor(probe, 'probe')
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError('Expected x[N,d], d>=1.')
    n = data_qubits(x.shape[1], encoding)
    if probe.ndim != 1 or probe.numel() != n:
        raise ValueError('Expected probe[n] matching the encoding register size.')
    if not isinstance(max_order, int) or isinstance(max_order, bool) or max_order < 0:
        raise ValueError('max_order must be a nonnegative integer.')
    if not isinstance(scale, numbers.Real) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be a positive finite real number.')
    if x.device != probe.device:
        raise ValueError('x and probe must use the same device.')
    _finite(x, 'x')
    _finite(probe, 'probe')
    if encoding in ('iqp', 'pauli'):
        return _root_moments(x*scale, probe, encoding, max_order)
    dtype = torch.promote_types(x.dtype, probe.dtype)
    x, probe = x.to(dtype)*scale, probe.to(dtype)
    orders = torch.arange(max_order+1, dtype=dtype, device=x.device)
    if encoding == 'angle':
        angles = orders[None, :, None]*x[:, None, :]/2
        return (torch.cos(angles)-1j*torch.sin(angles)*torch.sin(probe)).prod(dim=-1)
    # Padding follows scaling: the extra entries are zero phases, hence U=1.
    phases = torch.nn.functional.pad(x, (0, 2**n-x.shape[1]))
    weights = prepare_probe(probe).abs().square()
    return (torch.exp(1j*orders[None, :, None]*phases[:, None, :])*weights).sum(dim=-1)


def scores(x: torch.Tensor, theta: torch.Tensor, probe: torch.Tensor,
           encoding: str, scale: float = 1.) -> torch.Tensor:
    """Return processing-Z expectations for theta[T+1,3] and product Ry probe."""
    _real_tensor(theta, 'theta')
    _real_tensor(x, 'x')
    _real_tensor(probe, 'probe')
    if theta.ndim != 2 or theta.shape[0] < 1 or theta.shape[1] != 3:
        raise ValueError('Expected theta[T+1,3], T>=0.')
    if x.device != theta.device or probe.device != theta.device:
        raise ValueError('x, theta and probe must use the same device.')
    _finite(theta, 'theta')
    return scores_from_moments(moments(x, probe, encoding, theta.shape[0]-1, scale), theta)
