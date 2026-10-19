"""Exact differentiable m=1 QRENN for the A07 full-feature experiment.

Conventions (matching bundled QuAIRKit 0.4.4):
  B(theta) = exp(+i (theta[0] X + theta[1] Y + theta[2] Z)); no 1/2.
  |p> = tensor_j Ry(probe[j])|0>; processing qubit starts in |0>.
  Circuit time order: B[0], C(U), B[1], C(U†), ..., B[T].
  The oracle acts only on processing |1>. The score is <Z>; positive is label 1.
  Pauli U = tensor_j R_{X,Y,Z cyclic}(x[j]), R_P(a)=exp(-iaP/2).
  IQP U = H^n D H^n, D=exp[-i/2 (sum x_j Z_j +
                                  sum x_j x_{j+1} Z_j Z_{j+1})].

moments returns [N,K+1], mu[:,k]=<p|U**k|p> for nonnegative k.
Negative moments obey mu[-k]=conj(mu[k]). Frozen-probe training may compute
moments(..., max_order=T).detach() once and use scores_from_moments.
No dense oracle or exponentially large state is constructed by this module.
"""
from __future__ import annotations

import torch


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.complex128
    if dtype == torch.float32:
        return torch.complex64
    raise ValueError('Use real float32 or float64 parameters.')


def moments(x: torch.Tensor, probe: torch.Tensor, encoding: str, max_order: int) -> torch.Tensor:
    """Compute exact positive oracle moments in O(N*n*(max_order+1)).

    x has shape [N,n] and probe [n]. Gradients in both are retained. The IQP
    sum over the diagonal basis uses a two-state nearest-neighbor transfer
    recursion with weights (1 +/- sin(probe[j]))/2 after the first H layer.
    """
    if x.ndim != 2 or x.shape[1] < 1 or probe.ndim != 1 or probe.shape[0] != x.shape[1]:
        raise ValueError('Expected x[N,n] and probe[n], n>=1.')
    if not isinstance(max_order, int) or isinstance(max_order, bool) or max_order < 0:
        raise ValueError('max_order must be a nonnegative integer.')
    if encoding not in ('pauli', 'iqp'):
        raise ValueError('encoding must be pauli or iqp.')
    if x.device != probe.device:
        raise ValueError('x and probe must use the same device.')
    dtype = torch.promote_types(x.dtype, probe.dtype)
    cdtype = _complex_dtype(dtype)
    x, probe = x.to(dtype), probe.to(dtype)
    k = torch.arange(max_order+1, dtype=dtype, device=x.device)
    if encoding == 'pauli':
        axes = torch.arange(x.shape[1], device=x.device) % 3
        expectation = torch.where(axes == 0, torch.sin(probe),
                                  torch.where(axes == 2, torch.cos(probe), torch.zeros_like(probe)))
        angle = x[:, None, :] * k[None, :, None] / 2
        factors = torch.cos(angle).to(cdtype) - 1j * torch.sin(angle) * expectation
        return factors.prod(dim=-1)
    # s=+1,-1 is the Z eigenvalue; H Ry(b)|0> has this classical distribution.
    spins = torch.tensor([1., -1.], dtype=dtype, device=x.device)
    weights = (1 + torch.sin(probe)[:, None] * spins[None, :]) / 2
    unary = torch.exp(-0.5j * x[:, 0, None, None] * k[None, :, None] * spins)
    transfer = weights[0] * unary
    pair_spins = spins[:, None] * spins[None, :]
    for j in range(1, x.shape[1]):
        coupling = x[:, j-1]*x[:, j]
        edge = torch.exp(-0.5j * coupling[:, None, None, None] * k[None, :, None, None] * pair_spins)
        unary = torch.exp(-0.5j * x[:, j, None, None] * k[None, :, None] * spins)
        transfer = (transfer[:, :, :, None] * edge).sum(dim=-2) * weights[j] * unary
    return transfer.sum(dim=-1)


def _processing_coefficients(theta: torch.Tensor) -> torch.Tensor:
    """Amplitude coefficients c[p,a] of |p> tensor U**(a-T)|probe>."""
    if theta.ndim != 2 or theta.shape[1] != 3 or theta.shape[0] < 1:
        raise ValueError('Expected theta[T+1,3], T>=0.')
    cdtype = _complex_dtype(theta.dtype)
    paulis = torch.tensor([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]],
                         dtype=cdtype, device=theta.device)
    generators = torch.einsum('lk,kab->lab', theta.to(cdtype), paulis)
    blocks = torch.matrix_exp(1j * generators)
    t = theta.shape[0]-1
    initial = torch.zeros(2,2*t+1,dtype=cdtype,device=theta.device)
    initial[0,t] = 1
    coeff = initial
    for layer in range(t+1):
        coeff = blocks[layer] @ coeff
        if layer < t:
            branch = coeff[1]
            zero = torch.zeros_like(branch[:1])
            # Multiplying U^a by U shifts a to a+1, and U† shifts to a-1.
            branch = torch.cat([zero,branch[:-1]]) if layer % 2 == 0 else torch.cat([branch[1:],zero])
            coeff = torch.stack([coeff[0],branch])
    return coeff


def scores_from_moments(moment_array: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Contract fixed or differentiable positive moments with QRENN coefficients.

    moment_array is [N,K+1] and K>=T. Alternating query signs restrict the
    exponent support to an interval of length T+1, so moments through T suffice.
    Runtime is O(T^2 + N*T), independent of n after moment precomputation.
    """
    coeff = _processing_coefficients(theta)
    t = theta.shape[0]-1
    if moment_array.ndim != 2 or moment_array.shape[1] < t+1:
        raise ValueError('Expected moment_array[N,K+1] with K>=T.')
    if moment_array.device != theta.device:
        raise ValueError('moments and theta must use the same device.')
    dtype = torch.promote_types(moment_array.dtype, coeff.dtype)
    coeff = coeff.to(dtype)
    sign = torch.tensor([1,-1],dtype=dtype,device=theta.device)
    # <U^a p|U^b p> = mu_(b-a); negative differences are conjugate terms.
    diagonal = (sign * (coeff.conj()*coeff).sum(-1)).sum()
    value = diagonal * moment_array[:,0]
    for d in range(1,t+1):
        correlation = (sign[:,None] * coeff[:,:-d].conj() * coeff[:,d:]).sum()
        value = value + 2 * (correlation * moment_array[:,d]).real
    return value.real


def scores(x: torch.Tensor, theta: torch.Tensor, probe: torch.Tensor, encoding: str) -> torch.Tensor:
    """Batched processing-Z expectations; theta[T+1,3], probe[n], x[N,n]."""
    if theta.ndim != 2 or theta.shape[0] < 1 or theta.shape[1] != 3:
        raise ValueError('Expected theta[T+1,3], T>=0.')
    return scores_from_moments(moments(x,probe,encoding,theta.shape[0]-1), theta)
