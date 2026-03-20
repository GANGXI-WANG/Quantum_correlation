# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. Hardware Parameters (High-Res Idea B)
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

# Scale the participation matrix by Lamb-Dicke factor proportional to 1/sqrt(\omega)
b_eff_matrix = np.zeros_like(b_matrix)
for k in range(N_ions):
    b_eff_matrix[:, k] = b_matrix[:, k] * np.sqrt(f_com_kHz / f_k_array[k])

# ==========================================
# 2. Search for Optimal Interference Mode
# ==========================================
# We want to find a pair of ions (A, B) and a mode k where b_A,k ≈ - b_B,k
# This yields perfect destructive interference for the |eg> + |ge> state.

max_anti_ratio = 0
opt_A, opt_B, opt_k = -1, -1, -1

for i in range(N_ions):
    for j in range(i+1, N_ions):
        for k in range(N_ions):
            b_i = b_eff_matrix[i, k]
            b_j = b_eff_matrix[j, k]
            # We want them to have opposite signs and large magnitudes
            if b_i * b_j < 0:
                magnitude_product = np.abs(b_i * b_j)
                # Ensure they are also close in absolute value
                balance = min(np.abs(b_i), np.abs(b_j)) / max(np.abs(b_i), np.abs(b_j))
                score = magnitude_product * balance
                if score > max_anti_ratio:
                    max_anti_ratio = score
                    opt_A, opt_B, opt_k = i, j, k

print(f"Optimal Interference Pair: Ion {opt_A} & Ion {opt_B} at Mode {opt_k}")
print(f"b_A = {b_eff_matrix[opt_A, opt_k]:.4f}, b_B = {b_eff_matrix[opt_B, opt_k]:.4f}")

ion_A = opt_A
ion_B = opt_B
b_A_eff = b_eff_matrix[ion_A, :]
b_B_eff = b_eff_matrix[ion_B, :]

target_center = f_k_array[opt_k]
omega_k = 2 * np.pi * (f_k_array - target_center)
g = 2 * np.pi * 15.0 # 15 kHz for robust interference demonstration

# ==========================================
# 3. High-Res Dynamics
# ==========================================
def get_concurrence(P_eg, P_ge, P_ee, P_gg):
    return np.maximum(0, 2 * np.sqrt(P_eg * P_ge))

def get_dynamics_ideaB_highres(initial_sign, T_max=1.5):
    def odefun(t, y):
        c_eg, c_ge, c_k = y[0], y[1], y[2:]
        dy = np.zeros(2 + N_ions, dtype=np.complex128)

        amp = 1.0 # Single tone resonant with the optimal antisymmetric mode

        for k in range(N_ions):
            H_Ak = g * b_A_eff[k] * amp
            H_Bk = g * b_B_eff[k] * amp

            dy[0] += -1j * H_Ak * c_k[k]
            dy[1] += -1j * H_Bk * c_k[k]
            dy[k+2] += -1j * np.conj(H_Ak) * c_eg - 1j * np.conj(H_Bk) * c_ge - 1j * omega_k[k] * c_k[k]

        return dy

    y0 = np.zeros(2 + N_ions, dtype=np.complex128)
    y0[0] = 1.0 / np.sqrt(2)
    y0[1] = initial_sign * 1.0 / np.sqrt(2)

    t_eval = np.linspace(0, T_max, 2500)
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=t_eval, max_step=0.001, atol=1e-8, rtol=1e-8)

    P_eg = np.abs(sol.y[0])**2
    P_ge = np.abs(sol.y[1])**2
    P_gg = 1.0 - P_eg - P_ge
    P_ee = 0.0

    C = get_concurrence(P_eg, P_ge, P_ee, P_gg)
    return sol.t * 1000.0, C

t_us, C_sym = get_dynamics_ideaB_highres(1)  # Symmetric |eg> + |ge>
_, C_anti = get_dynamics_ideaB_highres(-1)   # Antisymmetric |eg> - |ge>

# Add a control group: destroy interference by breaking symmetry with a random bath
# We detune the drive slightly away from the resonant mode, or use two tones
# Let's use an off-resonant drive to show the standard decay
target_center_control = target_center + 5.0 # 5 kHz detuned
omega_k_control = 2 * np.pi * (f_k_array - target_center_control)
def get_dynamics_ideaB_control(initial_sign, T_max=1.5):
    def odefun(t, y):
        c_eg, c_ge, c_k = y[0], y[1], y[2:]
        dy = np.zeros(2 + N_ions, dtype=np.complex128)
        amp = 1.0
        for k in range(N_ions):
            H_Ak = g * b_A_eff[k] * amp
            H_Bk = g * b_B_eff[k] * amp
            dy[0] += -1j * H_Ak * c_k[k]
            dy[1] += -1j * H_Bk * c_k[k]
            dy[k+2] += -1j * np.conj(H_Ak) * c_eg - 1j * np.conj(H_Bk) * c_ge - 1j * omega_k_control[k] * c_k[k]
        return dy
    y0 = np.zeros(2 + N_ions, dtype=np.complex128)
    y0[0] = 1.0 / np.sqrt(2)
    y0[1] = initial_sign * 1.0 / np.sqrt(2)
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=np.linspace(0, T_max, 2500), max_step=0.001, atol=1e-8, rtol=1e-8)
    P_eg = np.abs(sol.y[0])**2
    P_ge = np.abs(sol.y[1])**2
    C = get_concurrence(P_eg, P_ge, 0.0, 1.0 - P_eg - P_ge)
    return sol.t * 1000.0, C

_, C_control = get_dynamics_ideaB_control(1)

# Save High-Res Data
np.savez('Data_IdeaB_DFS.npz', t_us=t_us, C_sym=C_sym, C_anti=C_anti, C_control=C_control)

# ==========================================
# Plot High-Res Idea B
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

ax.plot(t_us, C_sym, '-', lw=3.5, color='#16A085', label=r'对称态 $|\psi_+\rangle$ (相消干涉 / 无耗散保护)')
ax.plot(t_us, C_anti, '--', lw=3.0, color='#C0392B', label=r'反对称态 $|\psi_-\rangle$ (相长干涉 / 超辐射)')
ax.plot(t_us, C_control, ':', lw=2.5, color='#7F8C8D', label=r'对照组 (干涉破坏 / 常规耗散)')

# Annotations
text_str = f'Target Ions: {opt_A} and {opt_B}' + '\n' + f'Mode Frequency: {target_center:.1f} kHz'
ax.text(900, 0.4, text_str, fontsize=16, color='#1A252F',
        bbox=dict(facecolor='white', edgecolor='#BDC3C7', boxstyle='round,pad=0.5', alpha=0.9))

ax.set_xlabel('演化时间 (\mu s)', fontproperties='SimHei', fontsize=20, fontweight='bold')
ax.set_ylabel('两体关联度 (Concurrence)', fontproperties='SimHei', fontsize=20, fontweight='bold')
ax.set_title('晶格对称性诱导的无耗散子空间 (DFS)', fontproperties='SimHei', fontsize=22)
ax.set_xlim(0, 1500)
ax.set_ylim(-0.02, 1.05)
ax.legend(prop={'family': 'SimHei', 'size': 14}, loc='center right', frameon=False)

plt.tight_layout()
plt.savefig('Nature_IdeaB_HighRes_DFS.pdf', bbox_inches='tight')
plt.savefig('Nature_IdeaB_HighRes_DFS.png', bbox_inches='tight')
print("Idea B High-Res Data & Plot saved successfully.")