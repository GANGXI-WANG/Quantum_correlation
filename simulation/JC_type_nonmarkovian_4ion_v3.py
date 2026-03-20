# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. Load Experimental Data Dynamically
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Data", "data_transition_rsb_ion0of4_20260310_155809.npz")

data = np.load(file_path)
t_exp = data['x']        
y_exp = data['y'][:, 0]  

# ==========================================
# 2. Physical Parameters & SPAM Error
# ==========================================
f_k = np.array([1500.8, 1518.8, 1532.8, 1542.8]) 
f_com = 1542.8 
target_center = 1525.8 

omega_k = 2 * np.pi * (f_k - target_center) 
b_k = np.array([-0.19938972, -0.49318183, -0.68338526, -0.5])
b_k_eff = np.sqrt(f_com / (f_com + f_k)) * b_k

# Fitted effective Rabi frequency
g_real = 24.34 
g = 2 * np.pi * g_real

# 5-step preparation, 99% fidelity each
fidelity = 0.99 ** 5  # ~ 0.951
y_offset = 0.004      # Baseline dark state measurement error 

# ==========================================
# 3. Dynamics Evolution Solver
# ==========================================
def get_theoretical_dynamics(T_max):
    def odefun(t, y):
        c0, ck = y[0], y[1:]
        dy = np.zeros(5, dtype=np.complex128)
        
        for k in range(4):
            H_c = g * b_k_eff[k]
            dy[0] += -1j * H_c * ck[k]
            dy[k+1] += -1j * H_c * c0
        for k in range(4):
            dy[k+1] += -1j * omega_k[k] * ck[k]
        return dy

    y0 = np.zeros(5, dtype=np.complex128)
    y0[0] = 1.0 
    
    t_eval = np.linspace(0, T_max, 1500)
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=t_eval, max_step=0.001, atol=1e-8, rtol=1e-8)
    
    # Transition probability (P_D starting from empty state)
    P_transition = 1.0 - np.abs(sol.y[0])**2
    return sol.t, P_transition

T_max_ms = np.max(t_exp) / 1000.0
t_theory_ms, P_ideal = get_theoretical_dynamics(T_max_ms)

# Apply fidelity truncation and baseline offset
y_theory_measured = y_offset + fidelity * P_ideal
t_theory_us = t_theory_ms * 1000.0

# ==========================================
# 4. Nature-Style Figure Plotting (Optimized for 16-inch Screen)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',    
    'axes.unicode_minus': False,
    'font.size': 16,                # 放大基础字号，适配 16 英寸屏幕
    'axes.linewidth': 2.0,          # 加粗边框使其更具质感
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.direction': 'in',        
    'ytick.direction': 'in',
    'xtick.top': True,              
    'ytick.right': True
})

# dpi=120 用于您的屏幕预览尺寸，不会过大或过小
fig, ax = plt.subplots(figsize=(9, 6.5), dpi=120)

# Theory Curve (Crimson Red)
ax.plot(t_theory_us, y_theory_measured, '-', color='#C0392B', lw=3.0, zorder=1,
        label=r'Theory (with SPAM error)')

# Experimental Data (Sapphire Blue)
ax.plot(t_exp, y_exp, 'o', markersize=8.5, color='#2980B9', alpha=0.9, 
        markeredgecolor='#154360', markeredgewidth=1.2, zorder=2,
        label=r'Experiment (Ion 0)')

# Textbox with Physical Parameters
text_str = r'$\Omega_{\mathrm{eff}}/2\pi \approx 24.3$ kHz' + '\n' + r'$\mathcal{F}_{\mathrm{prep}} \approx 95.1\%$'
ax.text(180, 0.15, text_str, fontsize=15, color='#1A252F', 
        bbox=dict(facecolor='white', edgecolor='#BDC3C7', boxstyle='round,pad=0.5', alpha=0.9))

# Labels
ax.set_xlabel(r'Time ($\mu$s)', fontweight='bold', fontsize=18)
ax.set_ylabel(r'Transition probability', fontweight='bold', fontsize=18)

# Limits
ax.set_xlim(0, 310)
ax.set_ylim(-0.02, 1.05)

# Legend
ax.legend(loc='lower right', frameon=False, fontsize=15)

plt.tight_layout()

# 导出时强制使用 dpi=300，保证论文插入质量
plt.savefig('Nature_1Tone_Dynamics_Final.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Nature_1Tone_Dynamics_Final.png', dpi=300, bbox_inches='tight')
plt.show()