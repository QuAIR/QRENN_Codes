"""Exact fixed-basis states, local Pauli features, and classical kernels.

Angles are already scaled a(x) supplied by the caller: no fitting, scaling,
clipping, or other preprocessing occurs here. Processing qubits come first,
with qubit zero most significant. Public arrays return float64/complex128.
"""
from __future__ import annotations

import numbers

import numpy as np
import torch

from capacity_study.model import processing_gates

# Norm and expectation tolerances allow float64 contraction roundoff only.
_ROUNDOFF = 1e-10


def _array(value, name, *, complex_ok=False):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.requires_grad:
            raise ValueError(f'{name} must be a detached CPU tensor')
        try:
            value = value.numpy()
        except (TypeError, RuntimeError) as error:
            raise ValueError(f'{name} must have a supported numeric dtype') from error
    kinds = 'iufc' if complex_ok else 'iuf'
    if not isinstance(value, np.ndarray) or value.dtype.kind not in kinds:
        raise ValueError(f'{name} must be a real numeric array' if not complex_ok
                         else f'{name} must be a numeric array')
    value = np.asarray(value, dtype=np.complex128 if complex_ok else np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    return value


def _states(value, name):
    value = _array(value, name, complex_ok=True)
    if (value.ndim != 2 or value.shape[1] < 4
            or value.shape[1] & (value.shape[1]-1)):
        raise ValueError(f'{name} must have shape [N,2**q], q>=2')
    with np.errstate(over='ignore', invalid='ignore'):
        norms = np.sum(value.real**2+value.imag**2, axis=1)
    if not np.isfinite(norms).all() or np.any(abs(norms-1) > _ROUNDOFF):
        raise ValueError(f'{name} must contain normalized states')
    return value


def _features(value, name):
    value = _array(value, name)
    if value.ndim != 2 or value.shape[1] < 1:
        raise ValueError(f'{name} must have shape [N,p], p>=1')
    return value


def _positive(value, name):
    if (not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def _block_size(value):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise ValueError('block_size must be a positive integer')
    return int(value)


def feature_states(angles, theta, probe, *, n=5):
    """Return [N,4*2**n] output states; each preparation uses T queries.

    theta[T+1,15] uses the original SU4 Gell-Mann order and exp(+i H).
    The positive real input probe is sqrt(softmax([0, probe[D-1]])).
    Queries control branch |11>, alternating U then U-dagger, where U has
    phases exp(2*pi*i*pad(angles, 0, D-d)). No full-register gate is built.
    """
    if type(n) is not int or n < 1:
        raise ValueError('n must be a positive integer')
    angles = _array(angles, 'angles')
    theta = _array(theta, 'theta')
    probe = _array(probe, 'probe')
    dim = 2**n
    if angles.ndim != 2 or not 1 <= angles.shape[1] <= dim:
        raise ValueError('Expected angles[N,d], 1<=d<=2**n')
    if theta.ndim != 2 or theta.shape[0] < 1 or theta.shape[1] != 15:
        raise ValueError('Expected theta[T+1,15]')
    if probe.shape != (dim-1,):
        raise ValueError('Expected probe[2**n-1] logits')
    logits = np.r_[0., probe]
    # Large logit separations may underflow negligible probabilities to zero.
    with np.errstate(over='ignore', under='ignore'):
        weights = np.exp(logits-logits.max())
    psi = np.sqrt(weights/weights.sum())
    with np.errstate(over='ignore'):
        arguments = 2*np.pi*np.pad(angles, ((0, 0), (0, dim-angles.shape[1])))
    if not np.isfinite(arguments).all():
        raise ValueError('Phase arguments must be finite')
    phase = np.exp(1j*arguments)
    with torch.no_grad():
        blocks = processing_gates(torch.from_numpy(theta), m=2).numpy()
    if not np.isfinite(blocks).all():
        raise ValueError('Processing gates must be finite')
    state = np.zeros((len(angles), 4, dim), dtype=np.complex128)
    state[:, 0, :] = psi
    for layer, block in enumerate(blocks):
        state = np.einsum('pq,nqd->npd', block, state)
        if layer < len(blocks)-1:
            state[:, 3, :] *= phase if layer % 2 == 0 else phase.conj()
    result = state.reshape(len(angles), 4*dim)
    return _states(result, 'output states')


def projected_features(states):
    """Qubit-major X,Y,Z expectations, followed by processing Z0 Z1.

    Reshape pairs amplitudes on each bit; no full Pauli operator is built.
    Supplied states must already be normalized; no renormalization occurs.
    """
    states = _states(states, 'states')
    batch, width = states.shape
    qubits = width.bit_length()-1
    result = np.empty((batch, 3*qubits+1), dtype=np.float64)
    for qubit in range(qubits):
        paired = states.reshape(batch, 2**qubit, 2, 2**(qubits-qubit-1))
        zero, one = paired[:, :, 0, :], paired[:, :, 1, :]
        overlap = np.sum(zero.conj()*one, axis=(1, 2))
        result[:, 3*qubit] = 2*overlap.real
        result[:, 3*qubit+1] = 2*overlap.imag
        result[:, 3*qubit+2] = np.sum(abs(zero)**2-abs(one)**2, axis=(1, 2))
    probabilities = np.sum(abs(states.reshape(batch, 4, width//4))**2, axis=2)
    result[:, -1] = probabilities @ np.array([1., -1., -1., 1.])
    if not np.isfinite(result).all() or np.any(abs(result) > 1+_ROUNDOFF):
        raise ValueError('Pauli expectations exceeded physical bounds')
    return np.clip(result, -1, 1)


def fidelity_kernel(left_states, right_states, *, block_size=256):
    """Return exact |<left|right>|**2, in bounded overlap blocks.

    Each physical overlap contains two feature-state preparations (T+T
    queries). This is one kernel evaluation, not one preparation.
    """
    left = _states(left_states, 'left_states')
    right = _states(right_states, 'right_states')
    block_size = _block_size(block_size)
    if left.shape[1] != right.shape[1]:
        raise ValueError('State widths must match')
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for i in range(0, len(left), block_size):
        for j in range(0, len(right), block_size):
            values = abs(left[i:i+block_size].conj() @ right[j:j+block_size].T)**2
            if not np.isfinite(values).all() or np.any(values > 1+2*_ROUNDOFF):
                raise ValueError('Fidelity exceeded physical bounds')
            result[i:i+block_size, j:j+block_size] = np.clip(values, 0, 1)
    return result


def projected_gamma(fit_features, multiplier):
    """Fit-only gamma = multiplier/(p*Var(flattened fit features, ddof=0))."""
    fit = _features(fit_features, 'fit_features')
    multiplier = _positive(multiplier, 'multiplier')
    if len(fit) == 0:
        raise ValueError('fit_features must be nonempty')
    with np.errstate(over='ignore', invalid='ignore'):
        denominator = fit.shape[1]*np.var(fit, ddof=0)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Fit variance denominator must be positive and finite')
    with np.errstate(over='ignore', under='ignore'):
        gamma = multiplier/denominator
    return _positive(gamma, 'gamma')


def gaussian_kernel(left_features, right_features, gamma, *, block_size=256):
    """Gaussian Euclidean kernel with two-dimensional bounded temporaries.

    Accumulate coordinate differences directly to avoid cancellation from
    ||x||**2 + ||y||**2 - 2*x.y for nearby large coordinates.
    """
    left = _features(left_features, 'left_features')
    right = _features(right_features, 'right_features')
    gamma = _positive(gamma, 'gamma')
    block_size = _block_size(block_size)
    if left.shape[1] != right.shape[1]:
        raise ValueError('Feature widths must match')
    result = np.empty((len(left), len(right)), dtype=np.float64)
    for i in range(0, len(left), block_size):
        for j in range(0, len(right), block_size):
            a, b = left[i:i+block_size], right[j:j+block_size]
            distances = np.zeros((len(a), len(b)), dtype=np.float64)
            # Scale before squaring: d**2 can overflow even when gamma*d**2
            # is representable. Overflow after scaling has Gaussian zero.
            with np.errstate(over='ignore', under='ignore'):
                for coordinate in range(left.shape[1]):
                    difference = a[:, coordinate, None]-b[None, :, coordinate]
                    scaled = np.sqrt(gamma)*difference
                    distances += scaled*scaled
                values = np.exp(-distances)
            if not np.isfinite(values).all():
                raise ValueError('Gaussian kernel must be finite')
            result[i:i+block_size, j:j+block_size] = values
    return result
