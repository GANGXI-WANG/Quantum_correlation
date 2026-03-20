# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. Hardware Parameters (Idea 1 & 2 Base)
# ==========================================
# User provided 8-ion experimental parameters
dz_array = np.array([11.1628, 9.086, 8.6671, 8.6553, 8.6317, 9.0211, 10.6613])
f_com_kHz = 1506.8 # Using the highest mode as f_com

# Re-evaluating modes with collective_mode logic but using provided frequencies
# if possible. The user provided transverse mode frequencies:
# [1.5068 1.5038 1.4994 1.4938 1.4868 1.4794 1.4708 1.4614] (MHz) -> * 1000 for kHz
f_k_exp = np.array([1461.4, 1470.8, 1479.4, 1486.8, 1493.8, 1499.4, 1503.8, 1506.8])

# We use the collective_mode function to get the eigenvectors b_jk
def collective_mode(omx_rad, dz):
    num = len(dz) + 1
    z = np.zeros(num)
    for i in range(num - 1): z[i+1] = z[i] + dz[i]
    L = len(z)
    rinv3 = np.abs(np.reshape(z, (1, -1)) - np.reshape(z, (-1, 1)))
    np.fill_diagonal(rinv3, 1)
    rinv3 = 1 / rinv3**3
    np.fill_diagonal(rinv3, 0)
    coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / 1e3**2
    V = coef * rinv3
    V[range(L), range(L)] = -np.sum(V, axis=1) + omx_rad**2
    E, b_jk = np.linalg.eigh(V)
    return np.sqrt(np.maximum(E, 1e-10)), b_jk

omx_rad = 2 * np.pi * f_com_kHz
_, b_matrix = collective_mode(omx_rad, dz_array)
f_k_array = f_k_exp # Use experimental frequencies directly

N_ions = 8

# ==========================================
# Idea 1: Bandgap protection (Ion 0 & Ion 7 Entanglement)
# Low-Res Simulation
# ==========================================
# Select target ions for Bell state
ion_A = 0
ion_B = 7

b_A_eff = b_matrix[ion_A, :] * np.sqrt(f_com_kHz / f_k_array)
b_B_eff = b_matrix[ion_B, :] * np.sqrt(f_com_kHz / f_k_array)

# Find largest bandgap for protection
gaps = np.diff(f_k_array)
max_gap_idx = np.argmax(gaps)
gap_width = gaps[max_gap_idx]
target_center = f_k_array[max_gap_idx] + gap_width / 2.0

omega_k = 2 * np.pi * (f_k_array - target_center)
g = 2 * np.pi * 15.0 # We use 15 kHz coupling for low-res proof

def get_dynamics_idea1(drive_tones, T_max=0.4):
    """
    Simulate the two-ion system in the single-excitation subspace.
    Initial state: Bell state (|eg> + |ge>)/sqrt(2) with 0 phonons.
    State vector y:
    y[0]: c_eg (ion A excited, ion B ground, 0 phonons)
    y[1]: c_ge (ion A ground, ion B excited, 0 phonons)
    y[2:2+N_ions]: c_gg_k (both ground, 1 phonon in mode k)
    """
    def odefun(t, y):
        c_eg = y[0]
        c_ge = y[1]
        c_k = y[2:]
        dy = np.zeros(2 + N_ions, dtype=np.complex128)

        amp = sum(np.exp(-1j * 2 * np.pi * tune * t) for tune in drive_tones)
        amp = amp / np.sqrt(len(drive_tones)) # Power conservation

        for k in range(N_ions):
            H_Ak = g * b_A_eff[k] * amp
            H_Bk = g * b_B_eff[k] * amp

            # dot{c}_eg = -i \sum_k H_{Ak} c_k
            dy[0] += -1j * H_Ak * c_k[k]
            # dot{c}_ge = -i \sum_k H_{Bk} c_k
            dy[1] += -1j * H_Bk * c_k[k]

            # dot{c}_k = -i H_{Ak}^* c_eg -i H_{Bk}^* c_ge - i \omega_k c_k
            dy[k+2] += -1j * np.conj(H_Ak) * c_eg - 1j * np.conj(H_Bk) * c_ge - 1j * omega_k[k] * c_k[k]

        return dy

    y0 = np.zeros(2 + N_ions, dtype=np.complex128)
    y0[0] = 1.0 / np.sqrt(2)
    y0[1] = 1.0 / np.sqrt(2)

    t_eval = np.linspace(0, T_max, 500)
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=t_eval, max_step=0.005, atol=1e-6, rtol=1e-6)

    # Calculate C_zz = <sigma_z^A sigma_z^B> - <sigma_z^A><sigma_z^B>
    # For states restricted to |eg>, |ge>, |gg, 1_k>:
    # P_eg = |c_eg|^2, P_ge = |c_ge|^2, P_gg = \sum |c_k|^2 = 1 - P_eg - P_ge
    # sigma_z^A = P_eg - P_ge - P_gg = 2P_eg - 1
    # sigma_z^B = P_ge - P_eg - P_gg = 2P_ge - 1
    # <sigma_z^A sigma_z^B> = P_gg - P_eg - P_ge = P_gg - (1 - P_gg) = 2P_gg - 1

    P_eg = np.abs(sol.y[0])**2
    P_ge = np.abs(sol.y[1])**2
    P_gg = 1.0 - P_eg - P_ge

    sz_A = P_eg - P_ge - P_gg
    sz_B = P_ge - P_eg - P_gg
    sz_AB = P_gg - P_eg - P_ge

    C_zz = sz_AB - sz_A * sz_B
    return sol.t * 1000.0, C_zz

# Single tone inside bandgap (Non-Markovian memory)
t_us, Czz_1tone = get_dynamics_idea1([0.0])

# 5 tones covering the gap (Markovian continuum)
multi_tones = np.linspace(-gap_width/2.5, gap_width/2.5, 5)
t_us, Czz_5tone = get_dynamics_idea1(multi_tones)

# ==========================================
# Plot Idea 1
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(t_us, Czz_1tone, '-', lw=2, color='#C0392B', label='带隙内单频 (非马尔可夫保护)')
plt.plot(t_us, Czz_5tone, '--', lw=2, color='#2980B9', label='多频连续谱 (马尔可夫退相干)')
plt.xlabel('演化时间 (\mu s)')
plt.ylabel('两体关联度 C_{zz}')
plt.title('Idea 1：带隙内“拓扑保护”的长程纠缠演化 (低分辨率验证)', fontproperties='SimHei', fontsize=14)
plt.legend(prop={'family': 'SimHei', 'size': 12})
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('Idea1_LowRes_Proof.png', dpi=150)
print("Idea 1 Low-Res Proof saved.")
