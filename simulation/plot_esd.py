# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from single_double_sided import run_single_sided, run_double_sided

def plot_esd_comparison():
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

    # The prompt mentions:
    # "Target freq: 1463.0 kHz"
    # "Revival peak: t ~ 184 us"
    # To demonstrate ESD clearly, we probably want to look at a longer time or specific detuning.
    # The prompt mentions bandgap protection mode is 1463.0 kHz. Let's use this for both ESD and revival.

    T_max_us = 300 # Given in memory as ideal sim time
    target_freq = 1463.0

    print("Running Single-Sided...")
    t_s, P_s, C_s = run_single_sided(target_freq, T_max_us)

    print("Running Double-Sided... (This may take a minute)")
    t_d, P_d, C_d = run_double_sided(target_freq, T_max_us)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Subplot 1: Parity
    ax1 = axes[0]
    ax1.plot(t_d, P_d, '-', lw=3, color='#E74C3C', label='Double-Sided')
    ax1.plot(t_s, P_s, '--', lw=3, color='#3498DB', label='Single-Sided')
    ax1.axhline(0, color='gray', linestyle=':', lw=2)
    ax1.set_xlabel('Evolution Time ($\mu s$)', fontsize=20)
    ax1.set_ylabel('Parity $P(t)$', fontsize=20)
    ax1.set_title('Parity Evolution (ESD vs No ESD)', fontsize=22)
    ax1.legend(loc='best')

    # Subplot 2: Concurrence
    ax2 = axes[1]
    ax2.plot(t_d, C_d, '-', lw=3, color='#E74C3C', label='Double-Sided')
    ax2.plot(t_s, C_s, '--', lw=3, color='#3498DB', label='Single-Sided')

    # Also verify C(t) = sqrt(P(t)) for single sided
    ax2.plot(t_s, np.sqrt(np.maximum(0, P_s)), ':', lw=2, color='black', label='$\sqrt{P(t)}$ (Single)')

    ax2.axhline(0, color='gray', linestyle=':', lw=2)
    ax2.set_xlabel('Evolution Time ($\mu s$)', fontsize=20)
    ax2.set_ylabel('Concurrence $C(t)$', fontsize=20)
    ax2.set_title('Concurrence Dynamics', fontsize=22)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig('Nature_ESD_Comparison.pdf', bbox_inches='tight')
    plt.savefig('Nature_ESD_Comparison.png', bbox_inches='tight')
    print("ESD Comparison Plot saved successfully.")

if __name__ == "__main__":
    plot_esd_comparison()
