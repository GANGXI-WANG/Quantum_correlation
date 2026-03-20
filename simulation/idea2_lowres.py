# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# Hardware Parameters
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

# ==========================================
# Idea 2: Asymmetric structural bath
# Low-Res Simulation
# ==========================================
# Select target ions: Edge vs Center
ion_Edge = 0
ion_Center = 3

b_Edge_eff = b_matrix[ion_Edge, :] * np.sqrt(f_com_kHz / f_k_array)
b_Center_eff = b_matrix[ion_Center, :] * np.sqrt(f_com_kHz / f_k_array)

# We drive near the COM mode (where Center ion has high participation, Edge has low)
target_freq = f_k_array[-1] - 2.0 # 2 kHz red-detuned from COM mode
omega_k = 2 * np.pi * (f_k_array - target_freq)
g = 2 * np.pi * 12.0 # 12 kHz

def get_dynamics_idea2(drive_tones, T_max=0.4):
    def odefun(t, y):
        c_eg = y[0]
        c_ge = y[1]
        c_k = y[2:]
        dy = np.zeros(2 + N_ions, dtype=np.complex128)

        amp = sum(np.exp(-1j * 2 * np.pi * tune * t) for tune in drive_tones)
        amp = amp / np.sqrt(len(drive_tones))

        for k in range(N_ions):
            H_Ek = g * b_Edge_eff[k] * amp
            H_Ck = g * b_Center_eff[k] * amp

            dy[0] += -1j * H_Ek * c_k[k]
            dy[1] += -1j * H_Ck * c_k[k]
            dy[k+2] += -1j * np.conj(H_Ek) * c_eg - 1j * np.conj(H_Ck) * c_ge - 1j * omega_k[k] * c_k[k]

        return dy

    y0 = np.zeros(2 + N_ions, dtype=np.complex128)
    # Start with Bell state (|eg> + |ge>)/sqrt(2)
    y0[0] = 1.0 / np.sqrt(2)
    y0[1] = 1.0 / np.sqrt(2)

    t_eval = np.linspace(0, T_max, 500)
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=t_eval, max_step=0.005, atol=1e-6, rtol=1e-6)

    P_eg = np.abs(sol.y[0])**2  # Probability ion Edge is excited (information in Edge)
    P_ge = np.abs(sol.y[1])**2  # Probability ion Center is excited (information in Center)
    P_gg = 1.0 - P_eg - P_ge    # Information in the bath (phonons)

    return sol.t * 1000.0, P_eg, P_ge, P_gg

# Single tone driving
t_us, P_edge, P_center, P_bath = get_dynamics_idea2([0.0])

# ==========================================
# Plot Idea 2
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(t_us, P_edge, '-', lw=2, color='#E67E22', label='边缘离子 (Ion 0) 存活概率')
plt.plot(t_us, P_center, '--', lw=2, color='#8E44AD', label='中心离子 (Ion 3) 存活概率')
plt.plot(t_us, P_bath, ':', lw=2, color='#7F8C8D', label='热浴 (声子) 信息量')
plt.xlabel('演化时间 (\mu s)')
plt.ylabel('布居数概率')
plt.title('Idea 2：非对称结构化热浴中的信息定向泄露 (低分辨率验证)', fontproperties='SimHei', fontsize=14)
plt.legend(prop={'family': 'SimHei', 'size': 12})
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('Idea2_LowRes_Proof.png', dpi=150)
print("Idea 2 Low-Res Proof saved.")