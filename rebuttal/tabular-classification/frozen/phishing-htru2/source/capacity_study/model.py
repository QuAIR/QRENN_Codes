"""Exact Pauli moments with universal one/two-qubit processing contractions.

SU4 generator definition copied from universal_reference.py in
qrenn-universal-rebuttal-20261016 (basis function, row-major i,j; skip 0,0).
That reference repeats U. Here only the generator definition is reused:
queries alternate U,U-dagger, acting on the final processing branch only.
"""
from __future__ import annotations

import torch
from embedding_study.model import data_qubits, prepare_probe, moments, _real_tensor, _finite
from run import initial_parameters as _original_initial_parameters


def _validate_m(m):
    if type(m) is not int or m not in (1,2):
        raise ValueError('Processing qubits m must be 1 or 2')


def basis(m, *, dtype=torch.complex128, device='cpu'):
    """Trace-normalized XYZ (m=1) or exact reference Gell-Mann order (m=2)."""
    _validate_m(m)
    if m==1:
        return torch.tensor([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],dtype=dtype,device=device)
    matrices=[]
    for i in range(4):
        for j in range(4):
            if i==j==0:
                continue
            a=torch.zeros((4,4),dtype=dtype,device=device)
            if i==j:
                value=(2/(i*(i+1)))**.5
                a[torch.arange(i),torch.arange(i)]=value
                a[i,i]=-i*value
            elif i<j:
                a[i,j]=a[j,i]=1
            else:
                a[i,j],a[j,i]=1j,-1j
            matrices.append(a)
    return torch.stack(matrices)


def initial_parameters(n, slots, seed, m=1):
    """Match the original m1 probe draw at each seed,T across both m values."""
    _validate_m(m)
    theta,probe=_original_initial_parameters(n,slots,seed,'ry')
    if m==2:
        generator=torch.Generator(device='cpu').manual_seed(int(seed))
        theta=2*torch.pi*torch.rand((slots+1,15),generator=generator,dtype=torch.float64)
    return theta,probe


def processing_gates(theta, m=1):
    _validate_m(m)
    _real_tensor(theta,'theta')
    if theta.ndim!=2 or theta.shape[0]<1 or theta.shape[1]!=4**m-1:
        raise ValueError('Expected theta[T+1,4**m-1]')
    _finite(theta,'theta')
    cdtype=torch.complex128 if theta.dtype==torch.float64 else torch.complex64
    generators=torch.einsum('lk,kab->lab',theta.to(cdtype),basis(m,dtype=cdtype,device=theta.device))
    return torch.matrix_exp(1j*generators)


def processing_coefficients(theta, m=1):
    """c[p,a] represents |p> tensor U**(a-T)|probe>; queries shift p=2**m-1."""
    blocks=processing_gates(theta,m)
    T=theta.shape[0]-1
    coeff=torch.zeros((2**m,2*T+1),dtype=blocks.dtype,device=theta.device)
    coeff[0,T]=1
    for layer in range(T+1):
        coeff=blocks[layer]@coeff
        if layer<T:
            branch=coeff[-1]
            zero=torch.zeros_like(branch[:1])
            branch=torch.cat([zero,branch[:-1]]) if layer%2==0 else torch.cat([branch[1:],zero])
            coeff=torch.cat([coeff[:-1],branch[None,:]],dim=0)
    return coeff


def scores_from_moments(moment_array, theta, m=1):
    """Z tensor m parity contraction. Alternation limits exponent width to T."""
    coeff=processing_coefficients(theta,m)
    T=theta.shape[0]-1
    if moment_array.ndim!=2 or moment_array.shape[1]<T+1 or moment_array.device!=theta.device:
        raise ValueError('Expected same-device moments[N,K+1], K>=T')
    dtype=torch.promote_types(moment_array.dtype,coeff.dtype)
    coeff=coeff.to(dtype)
    sign=torch.tensor([(-1.)**k.bit_count() for k in range(2**m)],dtype=dtype,device=theta.device)
    value=(sign*(coeff.conj()*coeff).sum(-1)).sum()*moment_array[:,0]
    for delta in range(1,T+1):
        correlation=(sign[:,None]*coeff[:,:-delta].conj()*coeff[:,delta:]).sum()
        value=value+2*(correlation*moment_array[:,delta]).real
    return value.real


def scores(x, theta, probe, encoding='pauli', scale=1., m=1):
    if encoding!='pauli':
        raise ValueError('Capacity experiment permits the Pauli encoder only')
    if x.device!=theta.device or probe.device!=theta.device:
        raise ValueError('x,theta,probe must use the same device')
    return scores_from_moments(moments(x,probe,encoding,theta.shape[0]-1,scale),theta,m)
