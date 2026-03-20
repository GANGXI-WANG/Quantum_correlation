# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. Hardware Parameters (High-Res Idea A)
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

# Target symmetrical ions (Long-range Entanglement)
ion_A = 0
ion_B = 7

b_A_eff = b_matrix[ion_A, :] * np.sqrt(f_com_kHz / f_k_array)
b_B_eff = b_matrix[ion_B, :] * np.sqrt(f_com_kHz / f_k_array)

# Drive near the dense region for strong coupling
target_center = f_k_array[3] # 1486.8 kHz
omega_k = 2 * np.pi * (f_k_array - target_center)
g = 2 * np.pi * 24.34 # Very Strong coupling (24.34 kHz) for pronounced Sudden Death

def get_concurrence(P_eg, P_ge, P_ee, P_gg):
    return np.maximum(0, 2 * np.sqrt(P_eg * P_ge))

def get_dynamics_ideaA_highres(drive_tones, T_max=1.0):
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

    t_eval = np.linspace(0, T_max, 2500) # High resolution
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=t_eval, max_step=0.001, atol=1e-8, rtol=1e-8)

    P_eg = np.abs(sol.y[0])**2
    P_ge = np.abs(sol.y[1])**2
    P_gg = 1.0 - P_eg - P_ge
    P_ee = 0.0

    C = get_concurrence(P_eg, P_ge, P_ee, P_gg)
    return sol.t * 1000.0, C

# High-Res Simulation Execution
T_max_ms = 1.2
t_us, C_1tone = get_dynamics_ideaA_highres([0.0], T_max=T_max_ms)
_, C_3tone = get_dynamics_ideaA_highres([-4.5, 0.0, 4.5], T_max=T_max_ms)
_, C_5tone = get_dynamics_ideaA_highres([-9.0, -4.5, 0.0, 4.5, 9.0], T_max=T_max_ms)
_, C_7tone = get_dynamics_ideaA_highres([-13.5, -9.0, -4.5, 0.0, 4.5, 9.0, 13.5], T_max=T_max_ms)

# Save High-Res Data
np.savez('Data_IdeaA_SuddenDeath.npz', t_us=t_us, C_1tone=C_1tone, C_3tone=C_3tone, C_5tone=C_5tone, C_7tone=C_7tone)

# ==========================================
# Plot High-Res Idea A
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'font.size': 18,
    'axes.linewidth': 2.0,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

ax.plot(t_us, C_1tone, '-', lw=3.0, color='#C0392B', label='单频 (深非马尔可夫：纠缠复活)')
ax.plot(t_us, C_3tone, '--', lw=2.5, color='#8E44AD', label='三频 (有限带宽：微弱复活)')
ax.plot(t_us, C_5tone, '-.', lw=2.5, color='#2980B9', label='五频 (强耗散：单调纠缠猝死)')
ax.plot(t_us, C_7tone, ':', lw=2.5, color='#27AE60', label='七频 (极宽带：瞬时死亡)')

# Fill the zero-entanglement death regions for 1-tone to highlight revival
C_1tone_zero_mask = C_1tone < 1e-4
ax.fill_between(t_us, 0, 1.0, where=C_1tone_zero_mask, color='#F1C40F', alpha=0.15, transform=ax.get_xaxis_transform(), label='纠缠猝死暗区 (Sudden Death Region)')

# Annotations
text_str = r'Coupling $\Omega/2\pi \approx 24.3$ kHz' + '\n' + r'Target: Dense Region (1486.8 kHz)'
ax.text(800, 0.8, text_str, fontsize=16, color='#1A252F',
        bbox=dict(facecolor='white', edgecolor='#BDC3C7', boxstyle='round,pad=0.5', alpha=0.9))

ax.set_xlabel('演化时间 (\mu s)', fontproperties='SimHei', fontsize=20, fontweight='bold')
ax.set_ylabel('两体关联度 (Concurrence)', fontproperties='SimHei', fontsize=20, fontweight='bold')
ax.set_title('可编程热浴谱密度诱导的纠缠猝死相变', fontproperties='SimHei', fontsize=22)
ax.set_xlim(0, 1200)
ax.set_ylim(-0.02, 1.05)
ax.legend(prop={'family': 'SimHei', 'size': 14}, loc='upper right', frameon=False, bbox_to_anchor=(1.0, 0.75))

plt.tight_layout()
plt.savefig('Nature_IdeaA_HighRes_SuddenDeath.pdf', bbox_inches='tight')
plt.savefig('Nature_IdeaA_HighRes_SuddenDeath.png', bbox_inches='tight')
print("Idea A High-Res Data & Plot saved successfully.")