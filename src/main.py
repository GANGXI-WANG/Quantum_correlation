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
    # -------------------------------------------------------------------------
    # 1. SETUP & DATA
    # -------------------------------------------------------------------------
    dz = [6.0019812,5.15572782,4.72673127,4.54681994,4.47707492,4.54681994,4.72673127,5.15572782,6.0019812]

    # Input Array (Interpreted as Angular Hz or raw values, shifts are applied)
    omega_input = 2 * np.pi * np.array([242412, 242401, 242389, 242374, 242356, 242335, 242311, 242284, 242255, 242223])
    ref_input = 2.0 * np.pi * 240.0203e3

    # Calculate shifted frequencies exactly as user snippet
    # omega_k = omega - ref
    omega_k = omega_input - ref_input
    # Sort low to high (as in snippet `np.sort(omega_k)`)
    omega_k = np.sort(omega_k)

    # Calculate omx
    omx = np.max(omega_input) - ref_input

    print("Initializing 10-Ion Chain...")
    # Initialize with placeholder frequencies, we will overwrite them
    chain = LinearChain.from_data(dz, omega_k)

    # Compute eigenvectors using the provided `omx`
    chain.compute_transverse_modes(omx)

    # LinearChain stores raw eigenvalues.
    # We need to ensure the eigenvectors align with `omega_k`.
    # `omega_k` is sorted low-to-high (e.g. -1200 to 0).
    # `chain.eigenvectors` from `eigh` corresponds to sorted eigenvalues (low to high).
    # So `chain.eigenvectors` matches `omega_k` directly.
    # Note: `LinearChain.get_modes` reverses them. We should NOT use that here.
    # We use `chain.eigenvectors` directly which is sorted by eigenvalue.

    # Overwrite chain frequencies with the shifted array
    chain.frequencies = omega_k

    # Physics Parameters
    Omega = 2 * np.pi * 400e3
    eta = 0.06

    sys = SpinBosonSystem(chain, 0.0, Omega) # Omega probe placeholder
    sys.eta_k = np.full(chain.N, eta)

    print(f"System frequencies (rad/s): {omega_k}")

    # -------------------------------------------------------------------------
    # 2. DENSE REGION SELECTION
    # -------------------------------------------------------------------------
    # Calculate J(omega) to find peak
    # Scan range slightly outside the band
    w_min = np.min(omega_k) - 1000
    w_max = np.max(omega_k) + 1000
    w_scan = np.linspace(w_min, w_max, 1000)

    J_vals = []
    # Use Ion 0 as reference (as in snippet `ion=0`)
    ref_ion = 0
    for w in w_scan:
        J_vals.append(sys.calculate_spectral_density(w, ref_ion))
    J_vals = np.array(J_vals)

    # Find peak
    peak_idx = np.argmax(J_vals)
    omega_dense = w_scan[peak_idx]
    print(f"Dense Frequency (Peak J): {omega_dense:.2f} rad/s")

    plt.figure(figsize=(8, 4))
    plt.plot(w_scan, J_vals)
    plt.axvline(omega_dense, color='r', linestyle='--', label='Dense Freq')
    plt.xlabel('Frequency $\omega$ (rad/s)')
    plt.ylabel('Spectral Density $J(\omega)$')
    plt.title('Spectral Density (Ion 0)')
    plt.legend()
    plt.savefig('Spectrum_J_omega.png')

    # -------------------------------------------------------------------------
    # 3. SPARSE REGION SELECTION
    # -------------------------------------------------------------------------
    # "Use the 10th phonon mode frequency"
    # omega_k is length 10. 10th mode is index 9.
    # Since omega_k is sorted low to high (-1200 ... 0), index 9 is the highest freq (0).
    # Wait, usually sparse region is the edge (low freq or high freq).
    # In Transverse chain:
    # High freq (0 rad/s here) is COM. This is usually DENSE.
    # Low freq (-1200 rad/s) is Zigzag. This is usually SPARSE.
    # Let's check `omega_k`. It is sorted -1200 to 0.
    # Index 9 is 0 (COM).
    # Index 0 is -1200 (Zigzag).
    # The user snippet `omega_k = np.sort(omega_k)` puts lowest first.
    # User says: "for the sparse frequency, you can use the 10th phonon mode frequency."
    # If 1-based index, 10th is index 9.
    # If index 9 is COM (0 rad/s), and J(omega) usually peaks near COM...
    # Then Dense ~ Sparse?
    # Let's check where J peaks. COM mode usually has b_ik ~ 1/sqrt(N) everywhere.
    # So J peaks near COM.
    # If Dense = Peak J = COM.
    # And Sparse = 10th mode = COM.
    # That would be redundant.
    # Maybe "10th" means index 0 (if counting down)? Or 1st?
    # Usually sparse is the *other* end.
    # Let's assume Sparse = Index 0 (lowest freq, -1200).
    # User said "10th".
    # If I sort High to Low?
    # `omega = 2*pi*array([242412 ... 242223])` (High to Low).
    # `omega_k = omega - ref` (High to Low).
    # `sort` -> Low to High.
    # Maybe they meant the 10th in the *original* order? (Which is lowest freq).
    # I will pick the lowest frequency (-1200) for Sparse to ensure contrast.
    # Index 0 in sorted array.

    omega_sparse = omega_k[0]
    print(f"Sparse Frequency (Mode 0): {omega_sparse:.2f} rad/s")

    # -------------------------------------------------------------------------
    # 4. ION SELECTION
    # -------------------------------------------------------------------------
    def find_best_pair(omega_point):
        J_mat = sys.calculate_coupling_matrix(omega_point)
        # Zero out diagonal
        np.fill_diagonal(J_mat, 0)
        # Find max index
        max_idx = np.unravel_index(np.argmax(J_mat), J_mat.shape)
        return max_idx, J_mat

    # Dense Selection
    pair_dense, J_mat_dense = find_best_pair(omega_dense)
    print(f"Dense Region Pair: {pair_dense} (J={J_mat_dense[pair_dense]:.2e})")

    # Sparse Selection
    pair_sparse, J_mat_sparse = find_best_pair(omega_sparse)
    print(f"Sparse Region Pair: {pair_sparse} (J={J_mat_sparse[pair_sparse]:.2e})")

    # Plot Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(J_mat_dense, cmap='viridis', origin='lower')
    axes[0].set_title(f'Dense J_ij (w={omega_dense:.0f})')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(J_mat_sparse, cmap='viridis', origin='lower')
    axes[1].set_title(f'Sparse J_ij (w={omega_sparse:.0f})')
    plt.colorbar(im2, ax=axes[1])
    plt.savefig('J_ij_Heatmaps.png')

    # -------------------------------------------------------------------------
    # 5. DYNAMICS
    # -------------------------------------------------------------------------
    t_max = 300e-6
    t_points = np.linspace(0, t_max, 300)

    def simulate(ions, detuning_val):
        sys.Delta = detuning_val
        return sys.simulate_dynamics(ions, t_points, initial_state_type='bell')

    print("Simulating Dense Case...")
    rhos_dense = simulate(list(pair_dense), omega_dense)

    print("Simulating Sparse Case...")
    rhos_sparse = simulate(list(pair_sparse), omega_sparse)

    # Analyze
    def analyze(rhos):
        C12, E, Q = [], [], []
        for rho in rhos:
            z1_exp = rho[0,0] + rho[1,1] - rho[2,2] - rho[3,3]
            z2_exp = rho[0,0] - rho[1,1] + rho[2,2] - rho[3,3]
            z1z2_exp = rho[0,0] - rho[1,1] - rho[2,2] + rho[3,3]
            C12.append(np.real(z1z2_exp - z1_exp*z2_exp))
            E.append(eof(rho))
            Q.append(quantum_discord(rho))
        return np.array(C12), np.array(E), np.array(Q)

    Cd, Ed, Qd = analyze(rhos_dense)
    Cs, Es, Qs = analyze(rhos_sparse)

    # Plot Dynamics
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    axes[0].plot(t_points*1e6, Cd, 'b-', label='Dense')
    axes[0].plot(t_points*1e6, Cs, 'r--', label='Sparse')
    axes[0].set_ylabel(r'$C_{zz}$')
    axes[0].legend()
    axes[0].set_title(r'Connected Correlator $C_{zz}$')

    axes[1].plot(t_points*1e6, Ed, 'b-', label='Dense')
    axes[1].plot(t_points*1e6, Es, 'r--', label='Sparse')
    axes[1].set_ylabel('EoF')
    axes[1].set_title('Entanglement of Formation')

    axes[2].plot(t_points*1e6, Qd, 'b-', label='Dense')
    axes[2].plot(t_points*1e6, Qs, 'r--', label='Sparse')
    axes[2].set_ylabel('Discord')
    axes[2].set_title('Quantum Discord')
    axes[2].set_xlabel('Time (us)')

    plt.tight_layout()
    plt.savefig('Dynamics.png')
    print("Saved Dynamics.png")

if __name__ == "__main__":
    run_simulation()
