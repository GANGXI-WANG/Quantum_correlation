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

    # Frequency handling
    # The numbers in the array are treated as kHz.
    omegas_kHz_val = np.array([242412, 242401, 242389, 242374, 242356, 242335, 242311, 242284, 242255, 242223])
    ref_freq_kHz_val = 240020.3 # 240.0203e3

    # 1. Calculation of Eigenvectors (b_jk)
    # This must strictly follow the user's snippet logic where inputs are treated as is.
    # The snippet calculates omx using the raw values (interpreted as kHz? or Hz in that context?)
    # But to get the correct mode structure, we replicate the snippet's omx calculation.
    # Snippet: omega = 2*pi*array(...). omx = max(omega) - 2*pi*240020.3.
    # So we use the Angular Hz values of the raw inputs.
    omegas_raw_rad = 2 * np.pi * omegas_kHz_val
    ref_freq_raw_rad = 2 * np.pi * ref_freq_kHz_val
    omx_for_calc = np.max(omegas_raw_rad) - ref_freq_raw_rad

    print("Initializing 10-Ion Chain...")
    # Initialize chain with placeholders (frequencies will be overwritten)
    chain = LinearChain.from_data(dz, omegas_raw_rad)

    # Compute eigenvectors using the raw difference omx
    chain.compute_transverse_modes(omx_for_calc)
    _, evecs = chain.get_modes()
    chain.eigenvectors = evecs

    # 2. Calculation of Hamiltonian Frequencies
    # "frequency units is kHz" implies the difference (242412 - 240020 = 2392) is 2392 kHz = 2.392 MHz.
    # So we scale the difference by 1000 to get Hz, then 2*pi for Angular Hz.
    # difference_kHz = omegas_kHz_val - ref_freq_kHz_val
    # difference_Hz = difference_kHz * 1e3
    # omegas_hamiltonian_rad = 2 * np.pi * difference_Hz

    omegas_shifted_kHz = omegas_kHz_val - ref_freq_kHz_val
    omegas_hamiltonian_rad = 2 * np.pi * omegas_shifted_kHz * 1e3 # Convert kHz diff to Hz -> rad/s

    # Update chain frequencies for the physics module
    chain.frequencies = omegas_hamiltonian_rad

    # Simulation Parameters
    Omega = 2 * np.pi * 400e3 # 400 kHz -> 2*pi*400000 rad/s
    eta = 0.06

    # Heatmap
    w_center = np.mean(omegas_hamiltonian_rad)
    print("Generating J_ij Heatmap...")
    sys = SpinBosonSystem(chain, w_center, Omega)
    sys.eta_k = np.full(chain.N, eta)

    J = sys.calculate_coupling_matrix(w_center)

    plt.figure(figsize=(6, 5))
    plt.imshow(J, cmap='viridis', origin='lower')
    plt.title(r'Coupling Matrix $J_{ij}$ at Center of Band' + '\n' + r'(10 Ions, $\Omega$=400kHz, $\eta$=0.06)')
    plt.colorbar(label='Strength (rad/s?)')
    plt.xlabel('Ion Index')
    plt.ylabel('Ion Index')
    plt.savefig('J_ij_10ions.png')
    print("Saved J_ij_10ions.png")

    # Dynamics
    print("Simulating Dynamics...")
    ions = [4, 5]

    detuning_kHz = 0.0
    sys.Delta = omegas_hamiltonian_rad[0] + 2 * np.pi * detuning_kHz * 1e3

    # Physics parameters check
    coupling_g = eta * Omega / (2*np.pi) # in Hz
    bandwidth = (omegas_hamiltonian_rad[0] - omegas_hamiltonian_rad[-1]) / (2*np.pi) # in Hz
    mode_freq_MHz = w_center / (2*np.pi) / 1e6

    print(f"\n--- Simulation Parameters ---")
    print(f"Coupling strength g (eta*Omega): {coupling_g/1e3:.2f} kHz")
    print(f"Phonon Mode Center: {mode_freq_MHz:.4f} MHz")
    print(f"Phonon Bandwidth: {bandwidth/1e3:.2f} kHz")
    print(f"Ratio g / Bandwidth: {coupling_g/bandwidth:.2f}")
    print(f"Ratio g / ModeFreq: {coupling_g / (mode_freq_MHz*1e6):.4f}")
    print(f"Detuning from COM: {detuning_kHz:.2f} kHz")

    if coupling_g < bandwidth * 0.1:
         print("Regime: Weak Coupling (Markovian decay expected).")
    elif coupling_g > bandwidth * 10:
         print("Regime: Strong Coupling (Coherent oscillations).")
    else:
         print("Regime: Intermediate Coupling.")
    print("-----------------------------\n")

    # Print b_jk for user verification
    print("Coupling coefficients b_jk (for selected ions and top 3 modes):")
    for ion_idx in ions:
        print(f"Ion {ion_idx}: {chain.eigenvectors[ion_idx, :3]}")

    t_max = 300e-6 # 300 us
    t_points = np.linspace(0, t_max, 300)

    # Use Bell state initialization as requested
    rhos = sys.simulate_dynamics(ions, t_points, initial_state_type='bell')

    # Analyze
    C12_vals = []
    EoF_vals = []
    QD_vals = []

    for rho in rhos:
        # C12 = <z1 z2> - <z1><z2>
        z1_exp = rho[0,0] + rho[1,1] - rho[2,2] - rho[3,3]
        z2_exp = rho[0,0] - rho[1,1] + rho[2,2] - rho[3,3]
        z1z2_exp = rho[0,0] - rho[1,1] - rho[2,2] + rho[3,3]

        C12_vals.append(np.real(z1z2_exp - z1_exp*z2_exp))
        EoF_vals.append(eof(rho))
        QD_vals.append(quantum_discord(rho))

    # Plot Dynamics
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # 1. Connected Correlator (Measurable in Experiment)
    axes[0].plot(t_points*1e6, C12_vals, color='b', label=r'$C_{zz} = \langle \sigma_z^1 \sigma_z^2 \rangle - \langle \sigma_z^1 \rangle \langle \sigma_z^2 \rangle$')
    axes[0].set_ylabel(r'$C_{zz}$')
    axes[0].set_title(f'Experimentally Measurable Correlation ($C_{{zz}}$)\nResonant with COM ($\Delta = \omega_{{COM}}$)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Entanglement of Formation
    axes[1].plot(t_points*1e6, EoF_vals, color='g')
    axes[1].set_ylabel('Entanglement of Formation')
    axes[1].set_title('Quantum Entanglement')
    axes[1].grid(True, alpha=0.3)

    # 3. Quantum Discord
    axes[2].plot(t_points*1e6, QD_vals, color='r')
    axes[2].set_ylabel('Quantum Discord')
    axes[2].set_xlabel(r'Time ($\mu$s)')
    axes[2].set_title('Quantum Discord (Total Non-Classical Correlation)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Dynamics_10ions.png')
    print("Saved Dynamics_10ions.png")

if __name__ == "__main__":
    run_simulation()
