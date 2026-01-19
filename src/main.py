import numpy as np
import matplotlib.pyplot as plt
try:
    from src.ion_chain import LinearChain
    from src.physics import SpinBosonSystem
except ImportError:
    from ion_chain import LinearChain
    from physics import SpinBosonSystem
import os

def entropy(rho):
    """Von Neumann entropy."""
    evals = np.linalg.eigvalsh(rho)
    # Remove zeros/negatives for log
    evals = evals[evals > 1e-12]
    return -np.sum(evals * np.log2(evals))

def concurrence(rho):
    """
    Concurrence for 2 qubits.
    rho is 4x4 in basis ee, eg, ge, gg.
    """
    sy = np.array([[0, -1j], [1j, 0]])
    sy_sy = np.kron(sy, sy)

    rho_tilde = np.dot(np.dot(sy_sy, rho.conj()), sy_sy)

    R = np.dot(rho, rho_tilde)
    evals = np.sort(np.sqrt(np.abs(np.linalg.eigvals(R))))[::-1]

    c = evals[0] - evals[1] - evals[2] - evals[3]
    return max(0, c)

def eof(rho):
    """Entanglement of formation."""
    C = concurrence(rho)
    if C == 0:
        return 0
    if C >= 1: # Numerical errors
        return 1

    x = (1 + np.sqrt(1 - C**2)) / 2
    return -x * np.log2(x) - (1-x) * np.log2(1-x)

def quantum_discord(rho):
    """Quantum Discord."""
    rho_A = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
    rho_B = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)

    S_A = entropy(rho_A)
    S_B = entropy(rho_B)
    S_AB = entropy(rho)

    I_AB = S_A + S_B - S_AB

    min_conditional_entropy = 100

    thetas = np.linspace(0, np.pi, 10)
    phis = np.linspace(0, 2*np.pi, 10)

    I2 = np.eye(2)

    for theta in thetas:
        for phi in phis:
            nx = np.sin(theta) * np.cos(phi)
            ny = np.sin(theta) * np.sin(phi)
            nz = np.cos(theta)

            sigma_n = nx*np.array([[0,1],[1,0]]) + ny*np.array([[0,-1j],[1j,0]]) + nz*np.array([[1,0],[0,-1]])

            P0 = (I2 + sigma_n)/2
            P1 = (I2 - sigma_n)/2

            M0 = np.kron(I2, P0)
            M1 = np.kron(I2, P1)

            p0 = np.trace(np.dot(rho, M0)).real
            p1 = np.trace(np.dot(rho, M1)).real

            if p0 > 1e-10:
                rho_A_0 = np.trace(np.dot(M0, rho).reshape(2,2,2,2), axis1=1, axis2=3) / p0
                s0 = entropy(rho_A_0)
            else:
                s0 = 0

            if p1 > 1e-10:
                rho_A_1 = np.trace(np.dot(M1, rho).reshape(2,2,2,2), axis1=1, axis2=3) / p1
                s1 = entropy(rho_A_1)
            else:
                s1 = 0

            cond_ent = p0 * s0 + p1 * s1
            if cond_ent < min_conditional_entropy:
                min_conditional_entropy = cond_ent

    J_B = S_A - min_conditional_entropy
    QD = I_AB - J_B
    return max(0, QD)

def run_simulation():
    # 10-Ion Parameters provided by user
    dz = [6.0019812,5.15572782,4.72673127,4.54681994,4.47707492,4.54681994,4.72673127,5.15572782,6.0019812]

    # Frequency handling based on user clarification
    # Inputs are in kHz
    omegas_kHz_val = np.array([242412, 242401, 242389, 242374, 242356, 242335, 242311, 242284, 242255, 242223])
    ref_freq_kHz_val = 240020.3 # 240.0203e3

    # The shift determines the frame or effective frequencies
    # The physical Hamiltonian frequencies should be the difference (approx 2.4 MHz)
    # But we will convert everything to SI units (rad/s) for calculation

    omegas_Hz = omegas_kHz_val * 1e3
    ref_freq_Hz = ref_freq_kHz_val * 1e3

    # Shifted frequencies (Angular Hz)
    # User's omega_k = omega - shift
    omegas_shifted_rad = 2 * np.pi * (omegas_Hz - ref_freq_Hz)

    # Setup chain with these shifted frequencies as the Hamiltonian frequencies
    print("Initializing 10-Ion Chain...")
    print(f"Using Shifted Frequencies (e.g., Max: {np.max(omegas_shifted_rad)/(2*np.pi)/1e6:.4f} MHz)")
    chain = LinearChain.from_data(dz, omegas_shifted_rad)

    # 1. Calculation of Eigenvectors (b_jk)
    # We must calculate this using the RAW frequency scale (kHz inputs treated as is, resulting in small omx)
    # to be consistent with the user's snippet and hardcoded 'coef'.
    # If we use the MHz scale here, omx^2 becomes huge and dominates V, leading to localized modes (bad).

    # Use RAW angular frequencies for Mode Calculation
    omegas_raw_rad = 2 * np.pi * omegas_kHz_val
    ref_freq_raw_rad = 2 * np.pi * ref_freq_kHz_val
    omx_calc_rad = np.max(omegas_raw_rad) - ref_freq_raw_rad # Approx 2.4 kHz (angular)

    # Compute eigenvectors using the raw difference omx (Low Frequency Scale)
    chain.compute_transverse_modes(omx_calc_rad)
    _, evecs = chain.get_modes()
    chain.eigenvectors = evecs

    # 2. Setup Frequencies for Hamiltonian (Physical Scale)
    # The physical system should have bandwidth ~ MHz to match the weak coupling expectation.
    # User's "kHz units" comment implies the *difference* is in kHz -> scaled to MHz.
    # Or frequencies are MHz.
    # Current best guess: 242412 is kHz. Difference is 2.4 MHz.
    chain.frequencies = omegas_shifted_rad # This is the MHz scale (calculated above)

    # Simulation Parameters
    Omega = 2 * np.pi * 400e3 # 400 kHz -> 2*pi*400000 rad/s
    eta = 0.06

    # ---------------------------
    # New Plot 1: Mode Structure (b_ik)
    # ---------------------------
    plt.figure(figsize=(8, 6))
    # chain.eigenvectors is (N_ions, N_modes)
    # The provided omegas are sorted high to low.
    # Our get_modes() returns sorted high to low.
    # Mode 0 is COM (High freq). Mode 9 is Zigzag (Low freq).
    plt.imshow(chain.eigenvectors, cmap='coolwarm', aspect='auto', origin='upper')
    plt.colorbar(label='Amplitude $b_{ik}$')
    plt.xlabel('Mode Index (0=COM, High Freq)')
    plt.ylabel('Ion Index')
    plt.title('Transverse Mode Structure ($b_{ik}$)')
    plt.savefig('Mode_Structure_b_ik.png')
    print("Saved Mode_Structure_b_ik.png")

    # ---------------------------
    # New Plot 2: Spectrum
    # ---------------------------
    plt.figure(figsize=(8, 4))
    freqs_MHz = omegas_shifted_rad / (2*np.pi) / 1e6
    plt.stem(range(10), freqs_MHz)
    plt.xlabel('Mode Index')
    plt.ylabel('Frequency (MHz)')
    plt.title('Phonon Mode Spectrum')
    plt.grid(True, alpha=0.3)
    plt.savefig('Spectrum.png')
    print("Saved Spectrum.png")

    # Physics parameters check
    coupling_g = eta * Omega / (2*np.pi) # in Hz
    bandwidth = (omegas_shifted_rad[0] - omegas_shifted_rad[-1]) / (2*np.pi) # in Hz
    w_center = np.mean(omegas_shifted_rad)
    mode_freq_MHz = w_center / (2*np.pi) / 1e6

    print(f"\n--- Simulation Parameters ---")
    print(f"Coupling strength g (eta*Omega): {coupling_g/1e3:.2f} kHz")
    print(f"Phonon Mode Center: {mode_freq_MHz:.4f} MHz")
    print(f"Phonon Bandwidth: {bandwidth/1e3:.2f} kHz")
    print(f"Ratio g / Bandwidth: {coupling_g/bandwidth:.2f}")
    print(f"Ratio g / ModeFreq: {coupling_g / (mode_freq_MHz*1e6):.4f}")
    print("-----------------------------\n")

    sys = SpinBosonSystem(chain, w_center, Omega)
    sys.eta_k = np.full(chain.N, eta)
    ions = [4, 5]
    t_max = 300e-6
    t_points = np.linspace(0, t_max, 300)

    # Function to run simulation for a given detuning
    def simulate_scenario(detuning_kHz, label):
        print(f"Simulating Scenario: {label} (Detuning = {detuning_kHz:.2f} kHz)...")

        # J_ij for this scenario (evaluated at probe frequency)
        omega_probe = omegas_shifted_rad[0] + 2 * np.pi * detuning_kHz * 1e3

        # Calculate J
        # Update sys.Delta to match the probe/laser frame
        sys.Delta = omega_probe

        J = sys.calculate_coupling_matrix(omega_probe)

        plt.figure(figsize=(6, 5))
        plt.imshow(J, cmap='viridis', origin='lower')
        plt.title(rf'Coupling Matrix $J_{{ij}}$ ({label})' + '\n' + rf'($\Delta = {detuning_kHz:.1f}$ kHz)')
        plt.colorbar(label='Strength (rad/s)')
        plt.xlabel('Ion Index')
        plt.ylabel('Ion Index')
        plt.savefig(f'J_ij_{label}.png')
        print(f"Saved J_ij_{label}.png")

        # Dynamics
        rhos = sys.simulate_dynamics(ions, t_points, initial_state_type='bell')

        # Analyze
        C12 = []
        E = []
        Q = []
        for rho in rhos:
            z1_exp = rho[0,0] + rho[1,1] - rho[2,2] - rho[3,3]
            z2_exp = rho[0,0] - rho[1,1] + rho[2,2] - rho[3,3]
            z1z2_exp = rho[0,0] - rho[1,1] - rho[2,2] + rho[3,3]
            C12.append(np.real(z1z2_exp - z1_exp*z2_exp))
            E.append(eof(rho))
            Q.append(quantum_discord(rho))

        return np.array(C12), np.array(E), np.array(Q)

    # 1. Dense Region (Near COM / High Freq)
    # Detuning = 0 (Resonant with Mode 0 - Highest Freq)
    # The modes are denser near the top (differences ~11kHz) vs bottom (differences ~30kHz).
    C_dense, E_dense, Q_dense = simulate_scenario(0.0, "Dense")

    # 2. Sparse Region (Near Zigzag / Low Freq)
    # Detuning = Bandwidth (Resonant with Mode 9 - Lowest Freq)
    # omegas_shifted_rad[0] is High. omegas_shifted_rad[-1] is Low.
    # We want sys.Delta = omegas_shifted_rad[-1]
    # omegas_shifted_rad[-1] = omegas_shifted_rad[0] + 2*pi*diff
    # diff = (omegas[-1] - omegas[0])
    detuning_sparse_kHz = (omegas_shifted_rad[-1] - omegas_shifted_rad[0]) / (2*np.pi) / 1e3
    C_sparse, E_sparse, Q_sparse = simulate_scenario(detuning_sparse_kHz, "Sparse")

    # Comparison Plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # Czz
    axes[0].plot(t_points*1e6, C_dense, 'b-', label='Dense (COM)')
    axes[0].plot(t_points*1e6, C_sparse, 'r--', label='Sparse (Edge)')
    axes[0].set_ylabel(r'$C_{zz}$')
    axes[0].set_title(r'Experimentally Measurable Correlation ($C_{zz}$)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # EoF
    axes[1].plot(t_points*1e6, E_dense, 'b-', label='Dense')
    axes[1].plot(t_points*1e6, E_sparse, 'r--', label='Sparse')
    axes[1].set_ylabel('Entanglement of Formation')
    axes[1].set_title('Quantum Entanglement')
    axes[1].grid(True, alpha=0.3)

    # Discord
    axes[2].plot(t_points*1e6, Q_dense, 'b-', label='Dense')
    axes[2].plot(t_points*1e6, Q_sparse, 'r--', label='Sparse')
    axes[2].set_ylabel('Quantum Discord')
    axes[2].set_xlabel(r'Time ($\mu$s)')
    axes[2].set_title('Quantum Discord')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Dynamics_Comparison.png')
    print("Saved Dynamics_Comparison.png")

if __name__ == "__main__":
    run_simulation()
