# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. Hardware Parameters (Base for Idea A & B)
# ==========================================
dz_array = np.array([11.1628, 9.086, 8.6671, 8.6553, 8.6317, 9.0211, 10.6613])
f_com_kHz = 1506.8
f_k_exp = np.array([1461.4, 1470.8, 1479.4, 1486.8, 1493.8, 1499.4, 1503.8, 1506.8])

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
f_k_array = f_k_exp
N_ions = 8

# Target symmetrical ions
ion_A = 0
ion_B = 7

b_A_eff = b_matrix[ion_A, :] * np.sqrt(f_com_kHz / f_k_array)
b_B_eff = b_matrix[ion_B, :] * np.sqrt(f_com_kHz / f_k_array)

# We drive near the most dense region (strong coupling) to induce Sudden Death
target_center = f_k_array[3] # 1486.8 kHz
omega_k = 2 * np.pi * (f_k_array - target_center)
g = 2 * np.pi * 18.0 # Strong coupling (18 kHz) for low-res proof

def get_concurrence(P_eg, P_ge, P_ee, P_gg):
    """
    Calculate Entanglement Concurrence for X-state in subspace |ee>, |eg>, |ge>, |gg>
    Initial state: (|eg> + |ge>)/sqrt(2), so P_ee is always 0 in this 1-excitation subspace.
    C = max(0, 2*sqrt(P_eg*P_ge) - 2*sqrt(P_ee*P_gg))
    Since P_ee = 0, C = 2*sqrt(P_eg*P_ge)
    """
    return np.maximum(0, 2 * np.sqrt(P_eg * P_ge))

def get_dynamics_ideaA(drive_tones, T_max=0.3):
    """
    Simulate the two-ion system in the single-excitation subspace.
    Initial state: Bell state (|eg> + |ge>)/sqrt(2).
    """
    def odefun(t, y):
        c_eg, c_ge, c_k = y[0], y[1], y[2:]
        dy = np.zeros(2 + N_ions, dtype=np.complex128)

        amp = sum(np.exp(-1j * 2 * np.pi * tune * t) for tune in drive_tones)
        amp = amp / np.sqrt(len(drive_tones)) # Total power conservation

        for k in range(N_ions):
            H_Ak = g * b_A_eff[k] * amp
            H_Bk = g * b_B_eff[k] * amp

            dy[0] += -1j * H_Ak * c_k[k]
            dy[1] += -1j * H_Bk * c_k[k]
            dy[k+2] += -1j * np.conj(H_Ak) * c_eg - 1j * np.conj(H_Bk) * c_ge - 1j * omega_k[k] * c_k[k]

        return dy

    y0 = np.zeros(2 + N_ions, dtype=np.complex128)
    y0[0] = 1.0 / np.sqrt(2)
    y0[1] = 1.0 / np.sqrt(2)

    t_eval = np.linspace(0, T_max, 600)
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=t_eval, max_step=0.005, atol=1e-7, rtol=1e-7)

    P_eg = np.abs(sol.y[0])**2
    P_ge = np.abs(sol.y[1])**2
    P_gg = 1.0 - P_eg - P_ge # Phonons excited
    P_ee = 0.0 # Single excitation subspace

    C = get_concurrence(P_eg, P_ge, P_ee, P_gg)
    return sol.t * 1000.0, C

# Idea A: Broadening the bath spectrum to observe Transition
t_us, C_1tone = get_dynamics_ideaA([0.0])
t_us, C_3tone = get_dynamics_ideaA([-3.0, 0.0, 3.0])
t_us, C_5tone = get_dynamics_ideaA([-6.0, -3.0, 0.0, 3.0, 6.0])

# ==========================================
# Plot Idea A
# ==========================================
plt.rcParams.update({'font.family': 'sans-serif', 'axes.unicode_minus': False})
plt.figure(figsize=(8, 5))
plt.plot(t_us, C_1tone, '-', lw=2, color='#C0392B', label='1-Tone (窄带深非马尔可夫：纠缠猝死与多次复活)')
plt.plot(t_us, C_3tone, '--', lw=2, color='#8E44AD', label='3-Tone (中等带宽：衰减加快，微弱复活)')
plt.plot(t_us, C_5tone, ':', lw=2, color='#2980B9', label='5-Tone (宽带马尔可夫：彻底猝死无复活)')
plt.xlabel('演化时间 (\mu s)', fontsize=12)
plt.ylabel('两体纠缠度 (Concurrence)', fontsize=12)
plt.title('Idea A: 结构化热浴编程下的纠缠猝死与相变 (低分辨验证)', fontproperties='SimHei', fontsize=14)
plt.legend(prop={'family': 'SimHei', 'size': 10})
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('IdeaA_LowRes_Proof.png', dpi=150)
print("Idea A Low-Res Proof saved.")