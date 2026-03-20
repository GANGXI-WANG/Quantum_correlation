# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. 物理参数与离子链配置
# ==========================================
dz_array = np.array([10.5865, 9.0115, 8.4154, 8.2759, 8.4154, 9.0115, 10.5865]) 
f_com_kHz = 1506.8 
target_ion_idx = 0  

# ==========================================
# 2. 提取本征模式与环境拓扑寻优
# ==========================================
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
nu_kHz_rad, b_matrix = collective_mode(omx_rad, dz_array)

N_ions = len(dz_array) + 1 
f_k_array = nu_kHz_rad / (2 * np.pi) 

b_k = b_matrix[target_ion_idx, :]

# 【关键修正】：真正的 Lamb-Dicke 缩放
b_k_eff = b_k * np.sqrt(f_com_kHz / f_k_array)

# 真实的浴耦合权重 (您会发现它们恢复了正常高度)
J_heights = b_k_eff**2

# 自动寻找物理带隙
gaps = np.diff(f_k_array)
max_gap_idx = np.argmax(gaps)
gap_width = gaps[max_gap_idx]

# 将靶点精确锁定在带隙中心以产生 Yang 2013 的束缚态
target_center = f_k_array[max_gap_idx] + gap_width / 2.0 

# 三频间距恰好覆盖带隙以模拟连续谱耗散
custom_freq_shifts = [-gap_width/3, 0.0, gap_width/3]

# 恢复至弱耦合极限，完美呈现非马尔可夫震荡
g_val_kHz = 2.0 
g = 2 * np.pi * g_val_kHz
omega_k = 2 * np.pi * (f_k_array - target_center) 

fidelity = 0.99 ** 5  
y_offset = 0.004      
power_conservation = True 

# ==========================================
# 3. 动力学演化求解 (强耦合相变区)
# ==========================================
def get_dynamics(drive_tones, T_max=0.3):
    def odefun(t, y):
        c0, ck = y[0], y[1:]
        dy = np.zeros(N_ions + 1, dtype=np.complex128)
        
        amp = sum(np.exp(-1j * 2 * np.pi * tune * t) for tune in drive_tones)
        if power_conservation:
            amp = amp / np.sqrt(len(drive_tones))
            
        for k in range(N_ions):
            H_c = g * b_k_eff[k] * amp
            dy[0] += -1j * H_c * ck[k]
            dy[k+1] += -1j * np.conj(H_c) * c0
        for k in range(N_ions):
            dy[k+1] += -1j * omega_k[k] * ck[k]
        return dy

    y0 = np.zeros(N_ions + 1, dtype=np.complex128)
    y0[0] = 1.0 
    
    t_eval = np.linspace(0, T_max, 1500)
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=t_eval, max_step=0.001, atol=1e-8, rtol=1e-8)
    
    P_transition = 1.0 - np.abs(sol.y[0])**2
    return sol.t * 1000.0, y_offset + fidelity * P_transition

t_us, y_1tone = get_dynamics([0.0], T_max=0.5)
t_us, y_3tone = get_dynamics(custom_freq_shifts, T_max=0.5)

# ==========================================
# 4. 科技风制图 (纯中文，无标题，十三比十一)
# ==========================================
plt.rcParams.update({
    'font.family': ['SimHei', 'sans-serif'],
    'axes.unicode_minus': False,
    'font.size': 16,
    'axes.linewidth': 2.0,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11), dpi=150)

# ----- 上图: 频谱图与打靶位置 -----
ax1.axvspan(f_k_array[max_gap_idx], f_k_array[max_gap_idx+1], color='#F1C40F', alpha=0.15, label='物理带隙')
markerline, stemlines, baseline = ax1.stem(f_k_array, J_heights, basefmt=" ", bottom=0)
plt.setp(stemlines, color='#34495E', lw=2.5)
plt.setp(markerline, color='#2C3E50', markersize=8, label='离散热浴模式有效权重')

ax1.axvline(target_center, color='#16A085', linestyle='-', lw=2.5, alpha=0.8, label='单频驱动靶点（位于物理带隙）')
for i, shift in enumerate(custom_freq_shifts):
    label_str = '三频驱动梳齿（填平物理带隙）' if i == 0 else None
    ax1.axvline(target_center + shift, color='#C0392B', linestyle='--', lw=2.0, alpha=0.8, label=label_str)

ax1.set_xlabel('声子频率（千赫兹）', fontweight='bold')
ax1.set_ylabel('浴耦合权重（任意单位）', fontweight='bold')
ax1.set_xlim(np.min(f_k_array)-5, np.max(f_k_array)+5)
# 留出足够的空间展示恢复后的高峰
ax1.set_ylim(0, np.max(J_heights) * 1.3)
ax1.legend(loc='upper left', frameon=False, fontsize=14)

# ----- 下图: 相变演化对比 -----
ax2.plot(t_us, y_1tone, '-', color='#16A085', lw=3.0, label='单频驱动：带隙反弹（离子声子束缚态）')
ax2.plot(t_us, y_3tone, '-', color='#C0392B', lw=3.5, label='三频驱动：连续谱耗散（马尔可夫热化）')

text_str = f'寻址探测：零号离子\n等效拉比强度：{g_val_kHz} 千赫兹\n总制备保真度：约百分之九十五点一'
ax2.text(20, 0.15, text_str, fontsize=15, color='#1A252F', 
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.6', alpha=0.9))

ax2.set_xlabel('演化时间（微秒）', fontweight='bold')
ax2.set_ylabel('单离子跃迁概率', fontweight='bold')
ax2.set_xlim(0, 500)
ax2.set_ylim(-0.02, 1.05)
ax2.axhline(0.5, color='gray', linestyle=':', lw=2.0)
ax2.legend(loc='lower right', frameon=False, fontsize=14)

plt.tight_layout(pad=3.0)
plt.savefig('8Ion_PhaseTransition_CN.pdf', bbox_inches='tight')
plt.show()