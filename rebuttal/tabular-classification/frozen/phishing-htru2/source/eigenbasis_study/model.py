"""Exact paired diagonal eigenvalue model with unchanged universal SU(4) blocks.

Angles are the preprocessed a(x), not raw features. The formal configuration
has 30 angles, n=5, six complete Ry layers and five nearest-neighbor CZ chains.
Every input uses one fixed U=V D V-dagger throughout its controlled circuit.
Only sample-dependent phases and bases are cached; probe logits and processing
parameters remain differentiable and never enter prepare_inputs.
"""
from __future__ import annotations

import math
import numbers

import torch

from capacity_study.model import scores_from_moments

BASIS_MODES = ('fixed', 'input_dependent')


def _real64(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
        raise ValueError(f'{name} must be a float64 tensor')
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f'{name} must be finite')


def _register(features, n):
    if n is None:
        n = max(1, (features-1).bit_length())
    if type(n) is not int or n < 1 or features > 2**n:
        raise ValueError('n must be a positive integer with features <= 2**n')
    return n


def _basis(angles, n):
    """Build V in column-action convention, with qubit zero most significant."""
    batch, features = angles.shape
    dim = 2**n
    result = torch.eye(dim, dtype=torch.complex128, device=angles.device).expand(batch, dim, dim).clone()
    indices = torch.arange(dim, device=angles.device)
    parity = torch.zeros(dim, dtype=torch.int64, device=angles.device)
    for qubit in range(n-1):
        left = (indices >> (n-1-qubit)) & 1
        right = (indices >> (n-2-qubit)) & 1
        parity = parity ^ (left & right)
    cz = (1-2*parity).to(torch.float64)
    for layer in range(features//n):
        rotation = torch.ones((batch, 1, 1), dtype=torch.float64, device=angles.device)
        for qubit in range(n):
            half_angle = .5*torch.pi*(angles[:, layer*n+qubit]-.5)
            cosine, sine = half_angle.cos(), half_angle.sin()
            gate = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(batch, 2, 2)
            width = rotation.shape[-1]*2
            rotation = torch.einsum('nab,ncd->nacbd', rotation, gate).reshape(batch, width, width)
        result = rotation.to(torch.complex128) @ result
        if layer < features//n-1:
            result = cz[None, :, None]*result
    return result


@torch.no_grad()
def prepare_inputs(angles, basis_mode, T=8, scale=1.0, *, n=None):
    """Return same-device complex128 sample-major phase/basis tensors.

    phase_powers is [N,T+1,2**n]. input_dependent additionally supplies basis
    [N,2**n,2**n], representing V. Default n is the smallest positive register
    fitting all features. Explicit n supports independent smaller fixtures.
    The dependent mode requires complete rotation layers: features % n == 0.
    Scale multiplies a before both diagonal phases and Ry(pi*(a-.5)).
    """
    _real64(angles, 'angles')
    if angles.ndim != 2 or angles.shape[1] < 1:
        raise ValueError('Expected angles[N,features], features >= 1')
    if basis_mode not in BASIS_MODES:
        raise ValueError('basis_mode must be fixed or input_dependent')
    if type(T) is not int or T < 0:
        raise ValueError('T must be a nonnegative integer')
    if not isinstance(scale, numbers.Real) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be positive and finite')
    n = _register(angles.shape[1], n)
    if basis_mode == 'input_dependent' and angles.shape[1] % n:
        raise ValueError('input_dependent requires a complete number of n-qubit Ry layers')
    scaled = angles.detach()*scale
    if not bool(torch.isfinite(scaled).all()):
        raise ValueError('Scaled angles must be finite')
    phase = 2*torch.pi*torch.nn.functional.pad(scaled, (0, 2**n-scaled.shape[1]))
    orders = torch.arange(T+1, dtype=torch.float64, device=angles.device)
    arguments = orders[None, :, None]*phase[:, None, :]
    if not bool(torch.isfinite(arguments).all()):
        raise ValueError('Phase arguments must be finite')
    result = {'phase_powers': torch.exp(1j*arguments)}
    if basis_mode == 'input_dependent':
        result['basis'] = _basis(scaled, n)
    return result


def _prepared(prepared, *, check_finite=True):
    if not isinstance(prepared, dict) or 'phase_powers' not in prepared:
        raise ValueError('prepared must contain phase_powers')
    phases = prepared['phase_powers']
    if (not isinstance(phases, torch.Tensor) or phases.dtype != torch.complex128
            or phases.ndim != 3 or phases.shape[1] < 1 or phases.shape[2] < 2
            or phases.shape[2] & (phases.shape[2]-1)):
        raise ValueError('Expected complex128 phase_powers[N,T+1,2**n], n>=1')
    for name, value in prepared.items():
        if (not isinstance(value, torch.Tensor) or value.dtype not in (torch.float64, torch.complex128)
                or value.ndim < 1 or value.shape[0] != phases.shape[0] or value.device != phases.device):
            raise ValueError(f'{name} must be a same-device sample-major float64/complex128 tensor')
        if check_finite and not bool(torch.isfinite(value.detach()).all()):
            raise ValueError(f'{name} must be finite')
    if 'basis' in prepared:
        basis = prepared['basis']
        if basis.dtype != torch.complex128 or basis.shape != (phases.shape[0], phases.shape[2], phases.shape[2]):
            raise ValueError('Expected complex128 basis[N,D,D] matching phase_powers')
    return phases


def subset_inputs(prepared, index):
    """Check cache structure, then slice and validate only selected records.

    Nonfinite unselected rows do not invalidate an otherwise valid batch.
    Public score/weight consumers still fully validate their supplied batch.
    """
    _prepared(prepared, check_finite=False)
    if not isinstance(index, (slice, torch.Tensor, list, tuple)):
        raise ValueError('index must preserve the sample dimension')
    if isinstance(index, torch.Tensor) and index.ndim != 1:
        raise ValueError('Tensor index must be one-dimensional')
    result = {name: value[index] for name, value in prepared.items()}
    _prepared(result)
    return result


def _weights(prepared, probe, phases):
    _real64(probe, 'probe')
    if probe.ndim != 1 or probe.shape[0] != phases.shape[2]-1 or probe.device != phases.device:
        raise ValueError('Expected same-device probe[D-1] logits')
    logits = torch.cat((probe.new_zeros(1), probe))
    if 'basis' not in prepared:
        return torch.softmax(logits, dim=0).expand(phases.shape[0], -1)
    # exp(log_softmax/2) avoids undefined sqrt derivatives at underflowed zeros.
    psi = torch.exp(.5*torch.log_softmax(logits, dim=0)).to(torch.complex128)
    return (prepared['basis'].mH @ psi).abs().square()


def spectral_weights(prepared, probe):
    """Return differentiable float64 [N,D] weights |V-dagger psi|**2.

    psi is positive real in the computational basis. fixed mode has V=I and
    uses the original anchored-logit softmax exactly.
    """
    phases = _prepared(prepared)
    return _weights(prepared, probe, phases)


def scores_prepared(prepared, theta, probe):
    """Return float64 ZZ scores, controlled on |11>, alternating U/U-dagger."""
    phases = _prepared(prepared)
    _real64(theta, 'theta')
    if theta.shape != (phases.shape[1], 15) or theta.device != phases.device:
        raise ValueError('Expected same-device theta[T+1,15] matching cached phases')
    weights = _weights(prepared, probe, phases)
    moments = (phases*weights[:, None, :]).sum(-1)
    return scores_from_moments(moments, theta, m=2)


def scores(angles, theta, probe, basis_mode, scale=1.0, *, n=None):
    """Uncached inference wrapper; formal training should reuse prepared inputs."""
    _real64(theta, 'theta')
    if theta.ndim != 2 or theta.shape[0] < 1 or theta.shape[1] != 15:
        raise ValueError('Expected theta[T+1,15]')
    prepared = prepare_inputs(angles, basis_mode, theta.shape[0]-1, scale, n=n)
    return scores_prepared(prepared, theta, probe)
