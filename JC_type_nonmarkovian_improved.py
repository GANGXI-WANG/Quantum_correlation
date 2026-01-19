# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:39:44 2022
Non-Markovian dynamics of a spin coupled to multiple phonon modes
@author: wuyuk
"""
import numpy as np
from scipy.special import comb
from scipy.stats import geom
from scipy.sparse import csr_matrix, diags
from scipy.io import loadmat, savemat
from numba import jit
from qutip import Qobj, sesolve
import matplotlib.pyplot as plt
import os
import time

# data = loadmat('N20_spectrum.mat')
data = np.load("E:\\Gangxi\\20220818py-script\\experience_data\\20ions.12.22data_Load12023-12-23-14-43-07.npz")
L = data['L']
omx = data['omx']
print(omx)
omega_k = data['omega_k']
omega_k = omega_k - omx
b_jk = data['b_jk']
ion = 0
print(ion)
g = 2 * np.pi * 6.67
b_k = b_jk[ion, :]
print(b_k)
print(omega_k)
# nbar = 0 *np.ones(L)
nbar = 0.3 * np.ones(L)
nbar[[-1, -2, -3]] = [0.9, 0.5, 0.3]
mcut = 10 # mode cutoff, truncate to mcut closest modes
dimcut = 3e5 # dimension cutoff, if total dimension larger than this value, discard
M = 10 # sample for M times
T = 0.3 # ms, evolution time
step = 100
Delta = 2 * np.pi * -20 # kHz, detuning

# state1 = np.array([1, 0])
# state2 = np.array([0, 1])

state1 = 1/np.sqrt(2)*np.array([1, 1])
state2 = 1/np.sqrt(2)*np.array([1, -1])

if mcut > L:
    mcut = L
err = np.abs(g * b_k / (Delta - omega_k))
ind = np.argsort(err)[::-1][:mcut]
omega_k = omega_k[ind]
b_k = b_k[ind]
if len(nbar) == L:
    nbar = nbar[ind]
L = mcut
    
def DOS(omega_k, b_k, g):
    span = omega_k[-1] - omega_k[0]
    x = np.linspace(omega_k[0] - 0.5 * span, omega_k[-1] + 0.5 * span, 1000)
    y = 0
    for i in range(len(omega_k)):
        y = y + 1 / (1 + ((x - omega_k[i]) / b_k[i] / 2 / g)**2) \
                        * np.abs(2 * g * b_k[i])
    return(x, y)

# compute effective Hilbert space for sampled initial states
def sample_phonon_number(omx, omega_k, nbar):
    '''
    Randomly sample phonon number for all modes.
    omega_k is the actual mode frequency with omega_k[-1] == omx
    nbar is either the average phonon number for the freq. omx
        or an array for the average phonon number of each mode
    '''
    if len(nbar) == len(omega_k):
        nbar_k = nbar
    else:
        expinvT = (1 + 1 / nbar)**(omega_k / omx)
        nbar_k = 1 / (expinvT - 1)
    p = 1 / (nbar_k + 1) # parameter for geometric distribution
    n = geom.rvs(p, loc=-1, size=len(omega_k))
    return(n)
def construct_basis_boson(nph):
    '''
    Construct the basis states with a total phonon number of nph
    Return its dimension and the phonon number distribution for each state
    '''
    arr = [[L - 1 for i in range(nph)]] # where the excitations locate
    while nph > 0:
        i = nph - 1
        while (i > 0) and (arr[-1][i] == arr[-1][i - 1]):
            i -= 1
        if arr[-1][i] == 0:
            break
        arr.append(arr[-1][:i+1])
        arr[-1][i] -= 1
        arr[-1].extend([L - 1 for i in range(nph - i - 1)])
    l = len(arr)
    arr = np.array(arr, dtype=np.uint8)
    arr2 = np.zeros((l, L), dtype=np.uint8)
    for i in range(nph):
        arr2[range(l), arr[:, i]] += 1
    return(l, arr2)
def construct_basis_full(nex):
    '''
    Construct the basis states with a total excitation number of nex
    Return its dimension, the division of spin 0/1 subspaces,
    and the excitation number distribution for each state
    First site represents spin and the following L sites for phonon modes
    '''
    d1, basis1 = construct_basis_boson(nex)
    basis1 = np.hstack((np.zeros((d1, 1), dtype=np.uint8), basis1))
    if nex == 0:
        return(d1, d1, basis1)
    d2, basis2 = construct_basis_boson(nex - 1)
    basis2 = np.hstack((np.ones((d2, 1), dtype=np.uint8), basis2))
    d = d1 + d2
    basis = np.vstack((basis1, basis2))
    return(d, d1, basis)
@jit(nopython=True)
def compare_lessthan(arr1, arr2):
    for i in range(len(arr1)):
        if (arr1[i] < arr2[i]):
            return(True)
        elif (arr1[i] > arr2[i]):
            return(False)
    return(False)
@jit(nopython=True)
def search_index(arr, d, basis):
    l = 0
    r = d - 1
    while (l < r):
        mid = (l + r) // 2
        if compare_lessthan(basis[mid, :], arr):
            l = mid + 1
        else:
            r = mid
    return(l)
# construct Hamiltonian
@jit(nopython=True)
def construct_H_couple_entries(dim, basis, b_k):
    count = 0
    indptr = np.zeros(dim + 1, dtype=np.int64)
    for ind in range(dim):
        if basis[ind, 0] == 1:
            count += L
        else:
            count += np.sum(basis[ind, 1:] > 0)
        indptr[ind + 1] = count
    data = np.zeros(count, dtype=np.float64)
    indices = np.zeros(count, dtype=np.int64)
    for ind in range(dim):
        if basis[ind, 0] == 1:
            data[indptr[ind]:indptr[ind + 1]] = b_k * np.sqrt(basis[ind, 1:] 
                                                              + 1.0)
            state = np.copy(basis[ind, :])
            state[0] = 0
            for j in range(L):
                state[j + 1] += 1
                indices[indptr[ind] + j] = search_index(state, dim, basis)
                state[j + 1] -= 1
        else:
            sub = basis[ind, 1:] > 0
            nsub = np.sum(sub)
            indsub = np.arange(L)[sub]
            data[indptr[ind]:indptr[ind + 1]] = b_k[indsub] * np.sqrt(
                                            basis[ind, indsub + 1] + 0.0)
            state = np.copy(basis[ind, :])
            state[0] = 1
            for j in range(nsub):
                state[indsub[j] + 1] -= 1
                indices[indptr[ind] + j] = search_index(state, dim, basis)
                state[indsub[j] + 1] += 1
    return(data, indices, indptr)
def construct_H_couple(dim, basis, b_k):
    '''
    Construct Hamiltonian for the spin-phonon coupling
    '''
    data, indices, indptr = construct_H_couple_entries(dim, basis, b_k)
    H = csr_matrix((data, indices, indptr), shape=(dim, dim))
    return(H)
def construct_H_full(dim, basis, Delta, g, omega_k, b_k):
    '''
    Construct full Hamiltonian
    '''
    E = Delta * basis[:, 0] + basis[:, 1:] @ omega_k
    H = diags(E, format='csr') + g * construct_H_couple(dim, basis, b_k)
    return(H)
# time evolution and observables
def evaluate_rho(state, psi0, psi1, d0sub, d1sub):
    psi0 = psi0 / np.linalg.norm(psi0)
    psi1 = psi1 / np.linalg.norm(psi1)
    rho0 = np.array([[0, 0], [0, np.sum(np.abs(psi0[d0sub:])**2)]])
    psi0 = psi0[:d0sub]
    rho1 = np.array([[np.sum(np.abs(psi1[:d1sub])**2), 0], [0, 0]])
    psi1 = psi1[d1sub:]
    psi = np.array([state[0] * psi0, state[1] * psi1])
    rho = np.abs(state[0])**2 * rho0 + np.abs(state[1])**2 * rho1 \
        + psi @ psi.T.conj()
    return(rho / np.trace(rho))
def evolve_spin(state1, state2, n_k, T, step, Delta, g, omega_k, b_k):
    '''
    Evolve two spin states, state1 and state2, as well as initial phonon state
    n_k, for time T with $step$ steps.
    '''
    d0, d0sub, basis0 = construct_basis_full(np.sum(n_k))
    ind = search_index(np.concatenate(([0], n_k)), d0, basis0)
    psi0 = np.zeros(d0)
    psi0[ind] = 1
    H0 = construct_H_full(d0, basis0, Delta, g, omega_k, b_k)
    H0 = Qobj(H0)
    H0._type = 'oper'
    psi0 = Qobj(psi0)
    psi0._type = 'ket'
    d1, d1sub, basis1 = construct_basis_full(np.sum(n_k) + 1)
    ind = search_index(np.concatenate(([1], n_k)), d1, basis1)
    psi1 = np.zeros(d1)
    psi1[ind] = 1
    H1 = construct_H_full(d1, basis1, Delta, g, omega_k, b_k)
    H1 = Qobj(H1)
    H1._type = 'oper'
    psi1 = Qobj(psi1)
    psi1._type = 'ket'
    
    psi0_list = sesolve(H0, psi0, np.linspace(0, T, step + 1),
                        progress_bar=True).states
    psi1_list = sesolve(H1, psi1, np.linspace(0, T, step + 1),
                        progress_bar=True).states
    
    rho1_list = np.zeros((step + 1, 2, 2), dtype=complex)
    rho2_list = np.zeros((step + 1, 2, 2), dtype=complex)
    for i in range(step + 1):
        psi0 = psi0_list[i].data.toarray().flatten()
        psi1 = psi1_list[i].data.toarray().flatten()
        rho1_list[i, :, :] = evaluate_rho(state1, psi0, psi1, 
                                               d0sub, d1sub)
        rho2_list[i, :, :] = evaluate_rho(state2, psi0, psi1, 
                                               d0sub, d1sub)
    print(np.linalg.norm(psi0), np.linalg.norm(psi1))
    return(np.linspace(0, T, step + 1), rho1_list, rho2_list)

rho1_list = np.zeros((step + 1, 2, 2), dtype=complex)
rho2_list = np.zeros((step + 1, 2, 2), dtype=complex)
for it in range(M):
    n = sample_phonon_number(omx, omega_k + omx, nbar)
    d, _, _ = construct_basis_full(np.sum(n) + 1)
    if d > dimcut:
        print(it + 1, 'discarded')
        continue
    t_list, rho1, rho2 = evolve_spin(state1, state2, n, T, step, Delta, g,
                                     omega_k, b_k)
    rho1_list += rho1
    rho2_list += rho2
    print(it + 1)

def trace_dist(rho1, rho2):
    return(np.sum(np.linalg.svd(rho1 - rho2, compute_uv=False, 
                                hermitian=True)) / 2)
rho1_list /= M
rho2_list /= M
dist_list = np.zeros(step + 1)
for i in range(step + 1):
    dist_list[i] = trace_dist(rho1_list[i, :, :], rho2_list[i, :, :])

# temtime = time.strftime('%Y-%m-%d-%H-%M-%S')
# name = 'JC_type_nomarkovian_L{}_ion{}_Delta{:.2f}'.format(L, ion, Delta / 2 / np.pi) + str(temtime)
# BASE_PATH = "E:/Gangxi/20220818py-script/Simulation"
# np.savez(os.path.join(BASE_PATH, name), t_list = t_list, dist_list = dist_list, rho1_list=rho1_list, rho2_list=rho2_list)

plt.plot(t_list, dist_list, lw=3)
plt.xlabel('$t$/ms', fontsize=14)
plt.ylabel('Trace Distance', fontsize=14)
# plt.title(r'$L=20$, $d=6\,\mu$m, $|0\rangle$ vs $|1\rangle$, $\Delta/2\pi=-90$, $g/2\pi=10$, $\bar{n}_k=0.1$',
#           fontsize=14)
plt.show()