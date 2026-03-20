import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 用户控制台：核心参数与多频驱动设置
# ==========================================
# 填入你拟合好的离子间距 (um) 和质心频率 (kHz)
# 这里先用一组 8 离子的示例数据
dz_array = np.array([10.12, 8.87, 8.64, 8.65, 8.64, 8.87, 10.12]) 
f_com_kHz = 1506.8 

# --- 多频激光 (Multi-Tone) 设置 ---
# 1. 设置你想驱动的频率数量和相对失谐量 (单位: kHz)
# 例如 [-2.5, 0.0, 2.5] 代表三频，分别偏移 -2.5kHz, 0kHz, +2.5kHz
custom_freq_shifts = [-2.5, 0.0, 2.5] 

# 2. 是否开启“总功率守恒”？
# True  = 所有 Tone 平分总激光功率 (峰值和单频接近，用于对比噪音带形状)
# False = 每个 Tone 都满功率打入 (总耦合强度会成倍增加，如 3 Tone 约等于 3 倍强度)
power_conservation = False 

# ==========================================
# 2. 提取模式频率与本征矢量 b_jk
# ==========================================
def collective_mode(omx_kHz, dz):
    num = len(dz) + 1
    z = np.zeros(num)
    for i in range(num - 1):
        z[i+1] = z[i] + dz[i]
    L = len(z)
    rinv3 = np.abs(np.reshape(z, (1, -1)) - np.reshape(z, (-1, 1)))
    np.fill_diagonal(rinv3, 1) 
    rinv3 = 1 / rinv3**3
    np.fill_diagonal(rinv3, 0)
    coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / 1e3**2
    V = coef * rinv3
    V[range(L), range(L)] = -np.sum(V, axis=1) + omx_kHz**2
    E, b_jk = np.linalg.eigh(V)
    return np.sqrt(np.maximum(E, 1e-10)), b_jk

omx_rad = 2 * np.pi * f_com_kHz 
nu_kHz_rad, b_matrix = collective_mode(omx_rad, dz_array)

L = len(dz_array) + 1 # 离子总数
f_k_array = nu_kHz_rad / (2 * np.pi) # 转回纯频率 kHz
g_val = 2 * np.pi * 6.67 # 基准耦合

# ==========================================
# 3. 向量化的耦合强度计算公式
# ==========================================
def calc_coupling_strength(detuning_array, b_k_ion, f_k_arr, f_com, g, shift_frq=0):
    cs = np.zeros_like(detuning_array)
    for j in range(len(f_k_arr)):
        # 计算有效 Rabi 频率
        omega_eff = np.sqrt(f_com / (f_com + f_k_arr[j])) * (2 * g * b_k_ion[j])
        # 向量化计算 delta
        delta = 2 * np.pi * (f_k_arr[j] - (detuning_array - shift_frq))
        
        num = np.abs(omega_eff)**3
        den = omega_eff**2 + delta**2
        cs += np.abs(num / den)
    return cs

# ==========================================
# 4. 构造扫描区间与驱动参数
# ==========================================
k_points = 2000
detune = np.linspace(f_k_array[0] - 10, f_k_array[-1] + 10, k_points)

# 预分配存储空间
cs_single_all = np.zeros((L, k_points))
cs_multi_all  = np.zeros((L, k_points))

# 计算功率缩放因子
power_factor = len(custom_freq_shifts) if power_conservation else 1.0

# 遍历每个离子计算频谱
for ion_idx in range(L):
    b_k_ion = b_matrix[ion_idx, :] # 提取当前离子的本征矢量，表示该离子在各模式的参与度
    
    # 单频驱动 (Intrinsic)
    cs_single_all[ion_idx, :] = calc_coupling_strength(detune, b_k_ion, f_k_array, f_com_kHz, g_val, shift_frq=0)
    
    # 多频驱动叠加 (Engineered)
    for shift in custom_freq_shifts:
        cs_multi_all[ion_idx, :] += calc_coupling_strength(
            detune, b_k_ion, f_k_array, f_com_kHz, g_val, shift_frq=shift
        ) / power_factor

# ==========================================
# 5. 动态比例排版作图 (L行 x 2列)
# ==========================================
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
# 动态调整图片高度，确保无论多少个离子都不会挤在一起
fig, axs = plt.subplots(L, 2, figsize=(14, 2.5 * L), dpi=120, sharex=True)

# 统一 Y 轴量程，方便直观对比所有离子
y_max = max(np.max(cs_single_all), np.max(cs_multi_all)) * 1.1

for ion_idx in range(L):
    # --- 左列：单频激光 ---
    ax_left = axs[ion_idx, 0] if L > 1 else axs[0]
    ax_left.plot(detune, cs_single_all[ion_idx], color='#E64B35', lw=2, label=f'Ion {ion_idx} Intrinsic')
    ax_left.vlines(f_k_array, ymin=0, ymax=y_max, linestyles='dashed', colors='gray', alpha=0.4)
    ax_left.set_ylabel(f'Ion {ion_idx}\nCoupling', fontweight='bold')
    ax_left.set_ylim(0, y_max)
    ax_left.grid(True, linestyle=':', alpha=0.7)
    
    # 仅在首行和末行添加标题/标签
    if ion_idx == 0:
        ax_left.set_title('Intrinsic Spectrum (Single-Tone)', fontweight='bold')
    if ion_idx == L - 1:
        ax_left.set_xlabel('Detuning Frequency (kHz)', fontweight='bold')

    # --- 右列：多频激光 ---
    ax_right = axs[ion_idx, 1] if L > 1 else axs[1]
    ax_right.plot(detune, cs_multi_all[ion_idx], color='#3C5488', lw=2, label=f'Ion {ion_idx} Engineered')
    ax_right.fill_between(detune, 0, cs_multi_all[ion_idx], color='#3C5488', alpha=0.2)
    ax_right.vlines(f_k_array, ymin=0, ymax=y_max, linestyles='dashed', colors='gray', alpha=0.4)
    ax_right.set_ylim(0, y_max)
    ax_right.grid(True, linestyle=':', alpha=0.7)
    
    if ion_idx == 0:
        tone_str = f'{len(custom_freq_shifts)}-Tone Drive'
        ax_right.set_title(f'Engineered Spectrum ({tone_str})', fontweight='bold')
    if ion_idx == L - 1:
        ax_right.set_xlabel('Detuning Frequency (kHz)', fontweight='bold')

plt.tight_layout()
plt.show()