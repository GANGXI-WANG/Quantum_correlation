# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. Physical Parameters & SPAM
# ==========================================
f_k = np.array([1500.8, 1518.8, 1532.8, 1542.8]) 
f_com = 1542.8 
target_center = 1525.8 

omega_k = 2 * np.pi * (f_k - target_center) 
b_k = np.array([-0.19938972, -0.49318183, -0.68338526, -0.5])
b_k_eff = np.sqrt(f_com / (f_com + f_k)) * b_k

# Experimental SPAM Error 
fidelity = 0.99 ** 5  # ~ 0.951
y_offset = 0.004      

# ==========================================
# 2. Dynamics Solver (Weak Drive)
# ==========================================
def get_dynamics(g_real, drive_tones, T_max=0.45):
    g = 2 * np.pi * g_real
    def odefun(t, y):
        c0, ck = y[0], y[1:]
        dy = np.zeros(5, dtype=np.complex128)
        
        # Determine drive amplitude
        if len(drive_tones) == 1 and drive_tones[0] == 0:
            amp = 1.0
        else:
            amp = sum(np.exp(-1j * 2 * np.pi * tune * t) for tune in drive_tones)
            amp = amp / np.sqrt(len(drive_tones))
            
        for k in range(4):
            H_0k = g * b_k_eff[k] * amp
            dy[0] += -1j * H_0k * ck[k]
            dy[k+1] += -1j * np.conj(H_0k) * c0
            
        for k in range(4):
            dy[k+1] += -1j * omega_k[k] * ck[k]
        return dy

    y0 = np.zeros(5, dtype=np.complex128)
    y0[0] = 1.0 
    
    t_eval = np.linspace(0, T_max, 2000)
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=t_eval, max_step=0.001, atol=1e-8, rtol=1e-8)
    
    P_transition = 1.0 - np.abs(sol.y[0])**2
    return sol.t * 1000.0, y_offset + fidelity * P_transition

# 【核心参数注入】: 功率压低至 8.0 kHz 
g_weak = 8.0
t_us, y_1tone = get_dynamics(g_weak, [0.0], T_max=0.42)
t_us, y_3tone_incomm = get_dynamics(g_weak, [-4.5, 0.0, 3.14159], T_max=0.42)

# ==========================================
# 3. Nature-Style Plotting (Optimized for 16-inch)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',    
    'axes.unicode_minus': False,
    'font.size': 16,                
    'axes.linewidth': 2.0,          
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.direction': 'in',        
    'ytick.direction': 'in',
    'xtick.top': True,              
    'ytick.right': True
})

fig, ax = plt.subplots(figsize=(9, 6.5), dpi=120)

# Baseline: 1-Tone Protected Dynamics at weak driving
ax.plot(t_us, y_1tone, '--', color='#16A085', lw=3.0, zorder=1,
        label=r'1-Tone protected (8.0 kHz)')

# Bath Engineered: Incommensurate 3-Tone Thermalized Dynamics
ax.plot(t_us, y_3tone_incomm, '-', color='#E67E22', lw=3.5, zorder=2,
        label=r'3-Tone incommensurate (Markovian decay)')

# Textbox with parameters
text_str = r'Comb spacing: $\{-4.5, 0, +3.14\}$ kHz' + '\n' + r'$\Omega_{\mathrm{eff}}/2\pi \approx 8.0$ kHz'
ax.text(180, 0.15, text_str, fontsize=14, color='#1A252F', 
        bbox=dict(facecolor='white', edgecolor='#BDC3C7', boxstyle='round,pad=0.5', alpha=0.9))

# Labels and Setup
ax.set_xlabel(r'Time ($\mu$s)', fontweight='bold', fontsize=18)
ax.set_ylabel(r'Transition probability', fontweight='bold', fontsize=18)
ax.set_xlim(0, 420)
ax.set_ylim(-0.02, 1.05)
ax.axhline(0.5, color='gray', linestyle=':', lw=1.5, zorder=0)

ax.legend(loc='lower right', frameon=False, fontsize=13)

plt.tight_layout()

# Save for publication
plt.savefig('Nature_Markovian_Solution2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Nature_Markovian_Solution2.png', dpi=300, bbox_inches='tight')
plt.show()