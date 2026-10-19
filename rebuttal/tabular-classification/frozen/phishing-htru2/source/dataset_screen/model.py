"""Exact differentiable full-feature moments with unchanged SU(4) processing.

Qubit zero is most significant; processing starts |00>, controls |11>,
alternates U/U-dagger and reads ZZ. Positive scores favor label one.
No exponential data-register state is constructed for large product probes.
"""
from __future__ import annotations

import math
import numbers

import torch
from capacity_study.model import initial_parameters as _initial_parameters
from capacity_study.model import scores_from_moments
from model import moments as _root_moments

FAMILIES = ('angle_product', 'iqp_product', 'diagonal_product', 'diagonal_joint')


def _qubits(features, family):
    if type(features) is not int or features < 1 or family not in FAMILIES:
        raise ValueError('Positive integer features and a declared family required')
    return (features-1).bit_length() if family.startswith('diagonal') else features


def probe_size(features, family):
    """Number of free real probe parameters: n product angles or D-1 logits."""
    n = _qubits(features, family)
    return 2**n-1 if family == 'diagonal_joint' else n


def _real(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f'{name} must be a real float32/float64 tensor')
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f'{name} must be finite')


def _scale(scale):
    if not isinstance(scale, numbers.Real) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite scale required')


def _order(T):
    if type(T) is not int or T < 0:
        raise ValueError('Nonnegative integer query count required')


def _local_factors(probe):
    return torch.stack((torch.cos(probe/2), torch.sin(probe/2)), dim=1)


def _product_state(probe):
    state = torch.ones(1, dtype=probe.dtype, device=probe.device)
    for factor in _local_factors(probe):
        state = torch.kron(state, factor)
    return state


def _weights(probe, family):
    if family == 'diagonal_joint':
        return torch.softmax(torch.cat((probe.new_zeros(1), probe)), dim=0)
    return _product_state(probe).square()


def initial_parameters(features, family, T, seed):
    """Preserve original paired product draws and identical theta at seed/T.

    Joint diagonal logits are log(p[j]/p[0]) for the paired product weights;
    their first logit is fixed at zero and not a trainable parameter.
    """
    n = _qubits(features, family)
    _order(T)
    theta, probe = _initial_parameters(n, T, seed, m=2)
    if family == 'diagonal_joint':
        weights = _weights(probe, 'diagonal_product')
        if bool((weights <= 0).any()):
            raise ArithmeticError('Paired product initialization has zero probability')
        probe = torch.log(weights[1:])-torch.log(weights[0])
    return theta, probe


def _validate(angles, probe, family, T, scale):
    _real(angles, 'angles')
    _real(probe, 'probe')
    _scale(scale)
    _order(T)
    if angles.ndim != 2 or angles.shape[1] < 1:
        raise ValueError('Expected finite angles[N,d], d>=1')
    size = probe_size(angles.shape[1], family)
    if probe.ndim != 1 or probe.numel() != size:
        raise ValueError(f'Expected probe[{size}] for {family}')
    if angles.device != probe.device:
        raise ValueError('angles and probe must use the same device')


def moments(angles, probe, family, T, scale):
    """Complex [N,T+1] exact <p|U**k|p>, including controlled global phase."""
    _validate(angles, probe, family, T, scale)
    dtype = torch.promote_types(angles.dtype, probe.dtype)
    a, p = angles.to(dtype)*scale, probe.to(dtype)
    if family == 'iqp_product':
        return _root_moments(a, p, 'iqp', T)
    orders = torch.arange(T+1, dtype=dtype, device=a.device)
    if family == 'angle_product':
        phase = a[:, None, :]*orders[None, :, None]/2
        return (torch.cos(phase)-1j*torch.sin(phase)*torch.sin(p)).prod(dim=-1)
    n = _qubits(angles.shape[1], family)
    phase = 2*torch.pi*torch.nn.functional.pad(a, (0, 2**n-a.shape[1]))
    weights = _weights(p, family)
    if not bool(torch.isfinite(weights.detach()).all()) or not torch.allclose(
            weights.detach().sum(), weights.new_tensor(1.), atol=2e-6 if dtype == torch.float32 else 1e-12, rtol=0):
        raise ArithmeticError('Invalid physical probe normalization')
    return (torch.exp(1j*orders[None, :, None]*phase[:, None, :])*weights).sum(-1)


def scores(angles, theta, probe, family, scale):
    """Return [N] differentiable ZZ scores using universal SU(4) contraction."""
    _real(theta, 'theta')
    if theta.ndim != 2 or theta.shape[0] < 1 or theta.shape[1] != 15:
        raise ValueError('Expected theta[T+1,15]')
    if not isinstance(angles, torch.Tensor) or not isinstance(probe, torch.Tensor) or angles.device != theta.device or probe.device != theta.device:
        raise ValueError('angles, theta, probe must be tensors on the same device')
    return scores_from_moments(moments(angles, probe, family, theta.shape[0]-1, scale), theta, m=2)


def physical_probe(probe, features, family):
    """JSON-safe physical representative; large registers retain local factors.

    Product arrays are real two-amplitude factors (signs retained). Diagonal
    families also retain the small state; joint uses positive sqrt(weights).
    """
    _real(probe, 'probe')
    n = _qubits(features, family)
    if probe.ndim != 1 or probe.numel() != probe_size(features, family):
        raise ValueError('Probe shape does not match family/register')
    result = dict(family=family, features=features, qubits=n, basis_order='big_endian')
    if family == 'diagonal_joint':
        weights = _weights(probe, family)
        state = weights.sqrt()
        result.update(probabilities=weights.detach().cpu().tolist(), state=state.detach().cpu().tolist())
    else:
        factors = _local_factors(probe)
        if not torch.allclose(factors.detach().square().sum(-1), torch.ones(n, dtype=probe.dtype, device=probe.device), atol=2e-6 if probe.dtype == torch.float32 else 1e-12, rtol=0):
            raise ArithmeticError('Invalid local factor normalization')
        result['local_factors'] = factors.detach().cpu().tolist()
        if family.startswith('diagonal'):
            state = _product_state(probe)
            result.update(state=state.detach().cpu().tolist(), probabilities=state.square().detach().cpu().tolist())
    return result
