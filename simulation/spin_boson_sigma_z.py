# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. 物理参数与离子链配置 (Physical Parameters)
# ==========================================
# 离子间距，单位微米 (um)
dz_array = np.array([10.5865, 9.0115, 8.4154, 8.2759, 8.4154, 9.0115, 10.5865])
# 质心模式频率 f_com (kHz)
f_com_kHz = 1506.8
# 目标相互作用离子的索引（0 号离子）
target_ion_idx = 0

# ==========================================
# 2. 提取本征模式与环境参数计算
# ==========================================
def collective_mode(omx_rad, dz):
    """
    计算离子链的集体振动模式
    omx_rad: 质心模式圆频率
    dz: 离子间距数组
    返回: 模式圆频率 (nu_kHz_rad), 本征矩阵 (b_jk)
    """
    num = len(dz) + 1
    z = np.zeros(num)
    for i in range(num - 1): z[i+1] = z[i] + dz[i]
    L = len(z)
    rinv3 = np.abs(np.reshape(z, (1, -1)) - np.reshape(z, (-1, 1)))
    np.fill_diagonal(rinv3, 1)
    rinv3 = 1 / rinv3**3
    np.fill_diagonal(rinv3, 0)
    # 库仑力系数，适用于 171Yb+
    coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / 1e3**2
    V = coef * rinv3
    V[range(L), range(L)] = -np.sum(V, axis=1) + omx_rad**2
    E, b_jk = np.linalg.eigh(V)
    return np.sqrt(np.maximum(E, 1e-10)), b_jk

# 计算本征频率与本征向量
omx_rad = 2 * np.pi * f_com_kHz
nu_kHz_rad, b_matrix = collective_mode(omx_rad, dz_array)

N_ions = len(dz_array) + 1
f_k_array = nu_kHz_rad / (2 * np.pi)

# 获取靶离子的本征向量分量 b_k
b_k = b_matrix[target_ion_idx, :]

# 【关键修正】：真正的 Lamb-Dicke 缩放，引入 1/sqrt(\nu_k) 的依赖
b_k_eff = b_k * np.sqrt(f_com_kHz / f_k_array)

# 自动寻找物理带隙，确定激光靶点中心
gaps = np.diff(f_k_array)
max_gap_idx = np.argmax(gaps)
gap_width = gaps[max_gap_idx]
target_center = f_k_array[max_gap_idx] + gap_width / 2.0

# 计算各模式相对于激光失谐的圆频率
omega_k = 2 * np.pi * (f_k_array - target_center)

# ==========================================
# 3. 动力学演化参数与薛定谔方程求解
# ==========================================
# 设定的红边带耦合强度 g/2\pi = 24 kHz
g_val_kHz = 24.0
g = 2 * np.pi * g_val_kHz

# 实验 SPAM (State Preparation and Measurement) 误差设置
fidelity = 0.99 ** 5
y_offset = 0.004

def get_dynamics_sigma_z(T_max=0.3):
    """
    求解 Spin-Boson 模型的动力学演化。
    系统初态为离子处于激发态且声子处于真空态：|psi(0)> = |e, 0>
    演化过程中波函数展开为：|psi(t)> = c_0(t)|e, 0> + \\sum_k c_k(t)|g, 1_k>

    返回:
    t_us: 时间数组 (微秒)
    sigma_z: \\langle \\sigma_z(t) \\rangle 随时间的演化
    """
    def odefun(t, y):
        # y[0] 为 c_0，y[1:] 为 c_k
        c0, ck = y[0], y[1:]
        dy = np.zeros(N_ions + 1, dtype=np.complex128)

        # 旋转坐标系下，H_I = \sum_k (H_{0k} |e, 0><g, 1_k| + H_{0k}^* |g, 1_k><e, 0|)
        # 其中 H_{0k} = g * b_k_eff
        for k in range(N_ions):
            H_c = g * b_k_eff[k]
            # \dot{c}_0 = -i \sum_k H_{0k} c_k
            dy[0] += -1j * H_c * ck[k]
            # \dot{c}_k = -i H_{0k}^* c_0 - i \omega_k c_k
            dy[k+1] += -1j * np.conj(H_c) * c0
            dy[k+1] += -1j * omega_k[k] * ck[k]
        return dy

    # 初始状态：c_0(0) = 1，其他为 0
    y0 = np.zeros(N_ions + 1, dtype=np.complex128)
    y0[0] = 1.0

    # 演化时间采样点设置
    t_eval = np.linspace(0, T_max, 1500)
    sol = solve_ivp(odefun, [0, T_max], y0, t_eval=t_eval, max_step=0.001, atol=1e-8, rtol=1e-8)

    # 获取处在初态（即离子处于上能级，所有模式无声子）的概率 |c_0(t)|^2
    P_up = np.abs(sol.y[0])**2

    # 理论上的布居数差 <sigma_z> = P_up - P_down = 2 * P_up - 1
    # 考虑到测量中实验 SPAM 误差，计算带有误差的布居数差
    # 真实激发态分布 P_up_measured = y_offset + fidelity * P_up
    # 这里我们只展示考虑SPAM或理想情况下的 <sigma_z>。为严谨起见，这里按实验校准计算:
    # 实际处于下跌态的概率：P_transition = 1 - P_up
    # 测量到的态分布（假设 SPAM 只衰减了振幅和抬高基线）:
    P_down_measured = y_offset + fidelity * (1.0 - P_up)
    P_up_measured = 1.0 - P_down_measured

    # 绘制布居数差 <sigma_z> = P_up - P_down
    sigma_z = P_up_measured - P_down_measured

    # 我们也可以只绘制纯理论的 sigma_z，这里提供纯理论以供严格对比
    sigma_z_theory = 2.0 * P_up - 1.0

    return sol.t * 1000.0, sigma_z_theory, sigma_z

# 设定的最长演化时间为 300 微秒，因此传入 T_max = 0.3 ms
t_us, sigma_z_ideal, sigma_z_meas = get_dynamics_sigma_z(T_max=0.3)

# ==========================================
# 4. 数据可视化与绘图 (Nature-Style)
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

fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)

# 绘制 <sigma_z> 的演化曲线
ax.plot(t_us, sigma_z_ideal, '-', color='#C0392B', lw=3.0, zorder=1, label=r'Theory $\langle \sigma_z(t) \rangle$ (Ideal)')
ax.plot(t_us, sigma_z_meas, '--', color='#2980B9', lw=2.5, zorder=2, label=r'Theory $\langle \sigma_z(t) \rangle$ (with SPAM)')

# 标注物理参数的文本框
text_str = r'Coupling strength $\Omega/2\pi = 24.0$ kHz' + '\n' + r'Evolution time $t = 300\ \mu$s'
ax.text(180, 0.75, text_str, fontsize=15, color='#1A252F',
        bbox=dict(facecolor='white', edgecolor='#BDC3C7', boxstyle='round,pad=0.5', alpha=0.9))

# 设置横纵坐标轴与单位
ax.set_xlabel(r'Evolution Time ($\mu$s)', fontweight='bold', fontsize=18)
ax.set_ylabel(r'Population Difference $\langle \sigma_z(t) \rangle$', fontweight='bold', fontsize=18)

# 设置轴的显示范围
ax.set_xlim(0, 310)
ax.set_ylim(-1.05, 1.05)
ax.axhline(0, color='gray', linestyle=':', lw=1.5, zorder=0)

# 添加图例
ax.legend(loc='lower right', frameon=False, fontsize=14)

plt.tight_layout()

# 导出高质量图片
plt.savefig('Nature_SigmaZ_Evolution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Nature_SigmaZ_Evolution.png', dpi=300, bbox_inches='tight')
print("Simulation complete. Plot saved to 'Nature_SigmaZ_Evolution.png'")
