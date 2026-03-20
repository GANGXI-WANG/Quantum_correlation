# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理参数设定 (Physical Parameters)
# ==========================================
f_k = np.array([1500.8, 1518.8, 1532.8, 1542.8]) 
f_com = 1542.8 
target_center = 1525.8 

b_k = np.array([-0.19938972, -0.49318183, -0.68338526, -0.5])
b_k_eff = np.sqrt(f_com / (f_com + f_k)) * b_k

# 总等效拉比频率 (Fitted effective Rabi frequency)
g_real = 24.34 
J_discrete = (g_real * np.abs(b_k_eff))**2

# ==========================================
# 2. 【核心可调参数】多频激光的相对失谐 (kHz)
# ==========================================
# 您可以在这里任意调节 tune1 和 tune2 的值！
tune1 = -4.5 
tune2 = 4.5  
tones_delta = np.array([tune1, 0.0, tune2])

# 为了保证总激光功率不变，单束光的拉比频率减弱，功率展宽变窄
gamma_eff = g_real / np.sqrt(len(tones_delta))
J_discrete_eff = J_discrete / len(tones_delta)

# ==========================================
# 3. 计算有效谱密度 J_eff(w)
# ==========================================
omega = np.linspace(1480, 1565, 2000)
J_omega_total = np.zeros_like(omega)
J_components = []

# 计算 3 个 Tone 各自对应的平移谱密度
for delta in tones_delta:
    J_comp = np.zeros_like(omega)
    for i in range(4):
        # 旋转坐标系下的等效模式平移
        shifted_f = f_k + delta
        L_i = (gamma_eff/2)**2 / ((omega - shifted_f[i])**2 + (gamma_eff/2)**2)
        J_comp += J_discrete_eff[i] * L_i
    
    J_omega_total += J_comp
    J_components.append(J_comp)

# 归一化处理 (Normalized to arbitrary units)
norm_factor = np.max(J_omega_total)
J_omega_total /= norm_factor
for idx in range(len(J_components)):
    J_components[idx] /= norm_factor

# ==========================================
# 4. Nature-Style 科技风极简绘图
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

# 1. 标记原始带隙区域 (Golden shaded region)
ax.axvspan(1518.8, 1532.8, color='#F1C40F', alpha=0.15, zorder=0, label=r'Original bandgap')

# 2. 画出 3 个 Tone 分别的子谱密度包络 (Grey-blue dashed lines)
for i, J_comp in enumerate(J_components):
    label_str = r'Sub-spectra components' if i == 0 else None
    ax.plot(omega, J_comp, '--', color='#34495E', alpha=0.6, lw=1.5, zorder=2, label=label_str)

# 3. 画出叠加后的总有效谱密度 (Crimson Red solid line & fill)
ax.fill_between(omega, 0, J_omega_total, color='#C0392B', alpha=0.1, zorder=1)
ax.plot(omega, J_omega_total, '-', color='#C0392B', lw=3.5, zorder=3,
        label=r'Effective spectral density $J_{\mathrm{eff}}(\omega)$')

# 4. 标注激光的中心靶点 (Laser Target Center)
ax.axvline(target_center, color='#7F8C8D', linestyle=':', lw=2.0, zorder=1, label=r'Laser center (1525.8 kHz)')

# 坐标轴与范围设置
ax.set_xlabel(r'Phonon mode frequency (kHz)', fontweight='bold', fontsize=18)
ax.set_ylabel(r'Spectral density (a.u.)', fontweight='bold', fontsize=18)
ax.set_xlim(1485, 1560)
ax.set_ylim(0, 1.15)

# 图内学术参数文本框
text_str = r'Drive tones: $\{%.1f, 0, %.1f\}$ kHz' % (tune1, tune2) + '\n' + r'$\Omega_{\mathrm{total}}/2\pi \approx 24.3$ kHz'
ax.text(1488, 1.0, text_str, fontsize=14, color='#1A252F', 
        bbox=dict(facecolor='white', edgecolor='#BDC3C7', boxstyle='round,pad=0.5', alpha=0.9))

# 极简图例
ax.legend(loc='upper right', frameon=False, fontsize=13)

plt.tight_layout()

# 保存出版级 PDF 与预览 PNG
plt.savefig('Nature_Bath_Engineering_3Tone.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Nature_Bath_Engineering_3Tone.png', dpi=300, bbox_inches='tight')
plt.show()