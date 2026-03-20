# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from single_double_sided import run_single_sided

def plot_revival():
    # As requested: simulate Single-Sided at omega_L = 1463.0 kHz and
    # show the revival peak at t = 184 us where P ~ 0.926, C ~ 0.962.

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

    T_max_us = 300
    target_freq = 1463.0

    print("Running Single-Sided for Revival...")
    t_s, P_s, C_s = run_single_sided(target_freq, T_max_us)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    ax1.plot(t_s, P_s, '-', lw=3, color='#8E44AD', label='Parity $P(t)$')
    ax1.plot(t_s, C_s, '--', lw=3, color='#F39C12', label='Concurrence $C(t)$')

    # Mark the revival peak
    peak_idx = np.argmax(P_s[1000:]) + 1000 # Ignore the t=0 peak
    t_peak = t_s[peak_idx]
    P_peak = P_s[peak_idx]
    C_peak = C_s[peak_idx]

    ax1.plot(t_peak, P_peak, 'ro', markersize=10)

    text_str = f'Revival Peak\n$t \\approx {t_peak:.1f}$ $\\mu s$\n$P \\approx {P_peak:.3f}$\n$C \\approx {C_peak:.3f}$'
    ax1.annotate(text_str, xy=(t_peak, P_peak), xytext=(t_peak - 40, P_peak - 0.25),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 fontsize=16, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    ax1.set_xlabel('Evolution Time ($\\mu s$)', fontsize=20)
    ax1.set_ylabel('Amplitude', fontsize=20)
    ax1.set_title('Bandgap Protection Revival in Single-Sided Coupling', fontsize=22)
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(loc='lower right', frameon=False)

    # Add a text box about the bandgap
    bandgap_text = 'Laser Frequency: $\\omega_L = 1463.0$ kHz\n(Inside the phonon bandgap 1461.4 - 1470.8 kHz)'
    ax1.text(0.05, 0.25, bandgap_text, transform=ax1.transAxes, fontsize=16,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round,pad=0.4'))

    plt.tight_layout()
    plt.savefig('Nature_Revival_SingleSided.pdf', bbox_inches='tight')
    plt.savefig('Nature_Revival_SingleSided.png', bbox_inches='tight')
    print("Revival Plot saved successfully.")

if __name__ == "__main__":
    plot_revival()
