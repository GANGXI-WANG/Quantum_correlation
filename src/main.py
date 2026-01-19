import numpy as np
import matplotlib.pyplot as plt
from src.ion_chain import LinearChain
from src.physics import SpinBosonSystem
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
    # Spin flip operator Y x Y
    sy = np.array([[0, -1j], [1j, 0]])
    sy_sy = np.kron(sy, sy)

    rho_tilde = np.dot(np.dot(sy_sy, rho.conj()), sy_sy)

    # Eigenvalues of R = rho * rho_tilde
    # Or singular values of sqrt(rho) * sqrt(rho_tilde)
    # More stably: eigenvalues of R = sqrt(sqrt(rho) rho_tilde sqrt(rho))
    # But usually just eigenvalues of rho * rho_tilde is fine if rho is positive.

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
    """
    Quantum Discord QD(B|A) or A|B?
    Text says: QD_B(rho) = I(rho_AB) - max J_B(rho)
    J_B = S(A) - sum p_k S(A|k)
    Optimization over rank-1 POVM on B.
    """
    # Partial trace
    # Basis: ee(0), eg(1), ge(2), gg(3)
    # A is first qubit (rows 0,1 vs 2,3), B is second (cols 0,2 vs 1,3) inside blocks.
    # Actually, kron is usually A x B.
    # 0 (ee) -> 00
    # 1 (eg) -> 01
    # 2 (ge) -> 10
    # 3 (gg) -> 11

    rho_A = np.trace(rho.reshape(2,2,2,2), axis1=1, axis2=3)
    rho_B = np.trace(rho.reshape(2,2,2,2), axis1=0, axis2=2)

    S_A = entropy(rho_A)
    S_B = entropy(rho_B)
    S_AB = entropy(rho)

    I_AB = S_A + S_B - S_AB

    # Optimization over Projective Measurements on B
    # Direction n = (sin theta cos phi, sin theta sin phi, cos theta)
    # Projectors |+n><+n|, |-n><-n|

    # We scan theta, phi
    min_conditional_entropy = 100

    thetas = np.linspace(0, np.pi, 10)
    phis = np.linspace(0, 2*np.pi, 10)

    for theta in thetas:
        for phi in phis:
            nx = np.sin(theta) * np.cos(phi)
            ny = np.sin(theta) * np.sin(phi)
            nz = np.cos(theta)

            sigma_n = nx*np.array([[0,1],[1,0]]) + ny*np.array([[0,-1j],[1j,0]]) + nz*np.array([[1,0],[0,-1]])

            # Eigenvectors of n.sigma
            # easier: Projectors
            # P0 = (I + n.sigma)/2
            # P1 = (I - n.sigma)/2

            I2 = np.eye(2)
            P0 = (I2 + sigma_n)/2
            P1 = (I2 - sigma_n)/2

            # Measurement on B: I x P0, I x P1
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
    return max(0, QD) # Can be negative due to numerics, clamp to 0

def run_simulation():
    # Parameters
    N = 20
    f_min = 2.133e6
    f_max = 2.406e6

    omega_dense = 2.39569e6
    omega_sparse = 2.17681e6

    Omega = 10e3 # 10 kHz Rabi frequency (adjust to see dynamics)
    # The text mentions max J ~ 10^4.
    # J ~ eta^2 Omega.
    # If eta ~ 0.1, Omega ~ 1 MHz => J ~ 10 kHz.

    print("Initializing Ion Chain...")
    chain = LinearChain(N=N, alpha=0.002) # Adjusted alpha
    chain.compute_transverse_modes()
    chain.get_scaled_modes(f_min, f_max)

    # 1. Heatmaps
    print("Generating J_ij Heatmaps...")

    sys_dense = SpinBosonSystem(chain, omega_dense, Omega)
    sys_sparse = SpinBosonSystem(chain, omega_sparse, Omega)

    J_dense = sys_dense.calculate_coupling_matrix(omega_dense)
    J_sparse = sys_sparse.calculate_coupling_matrix(omega_sparse)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(J_dense, cmap='viridis', origin='lower')
    axes[0].set_title(f'Dense Regime ({(omega_dense/1e6):.4f} MHz)')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(J_sparse, cmap='viridis', origin='lower')
    axes[1].set_title(f'Sparse Regime ({(omega_sparse/1e6):.4f} MHz)')
    plt.colorbar(im2, ax=axes[1])

    plt.savefig('J_ij_heatmaps.png')
    print("Saved J_ij_heatmaps.png")

    # 2. Dynamics
    print("Simulating Dynamics...")
    # Center ions
    ions = [9, 10]

    # Time evolution
    t_max = 200e-6 # 200 us
    t_points = np.linspace(0, t_max, 100)

    # We need to set Delta such that we are probing the modes.
    # Usually Delta approx omega_phonon.
    # The text says "laser detuning... Delta".
    # And H = Delta/2 sigma_z + ...
    # If we want resonance, Delta ~ omega_k.
    # So we set Delta = omega_dense for dense case, etc.

    sys_dense.Delta = omega_dense
    sys_sparse.Delta = omega_sparse

    rhos_dense = sys_dense.simulate_dynamics(ions, t_points)
    rhos_sparse = sys_sparse.simulate_dynamics(ions, t_points)

    # Calculate Observables
    def analyze(rhos):
        C12 = []
        EoF_vals = []
        QD_vals = []
        for rho in rhos:
            # C12 = <z1 z2> - <z1><z2>
            # z1 = |e><e| - |g><g| tensor I
            # z2 = I tensor |e><e| - |g><g|
            # Basis: ee, eg, ge, gg
            # z1: 1, 1, -1, -1
            # z2: 1, -1, 1, -1
            # z1z2: 1, -1, -1, 1

            z1_exp = rho[0,0] + rho[1,1] - rho[2,2] - rho[3,3]
            z2_exp = rho[0,0] - rho[1,1] + rho[2,2] - rho[3,3]
            z1z2_exp = rho[0,0] - rho[1,1] - rho[2,2] + rho[3,3]

            C12.append(np.real(z1z2_exp - z1_exp*z2_exp))
            EoF_vals.append(eof(rho))
            QD_vals.append(quantum_discord(rho))
        return C12, EoF_vals, QD_vals

    C_d, E_d, Q_d = analyze(rhos_dense)
    C_s, E_s, Q_s = analyze(rhos_sparse)

    # Plot Dynamics
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    axes[0].plot(t_points*1e6, C_d, label='Dense', color='r')
    axes[0].plot(t_points*1e6, C_s, label='Sparse', color='b')
    axes[0].set_ylabel('C_12')
    axes[0].legend()
    axes[0].set_title('Connected Correlator')

    axes[1].plot(t_points*1e6, E_d, label='Dense', color='r')
    axes[1].plot(t_points*1e6, E_s, label='Sparse', color='b')
    axes[1].set_ylabel('Entanglement of Formation')
    axes[1].set_title('Entanglement')

    axes[2].plot(t_points*1e6, Q_d, label='Dense', color='r')
    axes[2].plot(t_points*1e6, Q_s, label='Sparse', color='b')
    axes[2].set_ylabel('Quantum Discord')
    axes[2].set_xlabel('Time (us)')
    axes[2].set_title('Quantum Discord')

    plt.tight_layout()
    plt.savefig('Dynamics.png')
    print("Saved Dynamics.png")

if __name__ == "__main__":
    run_simulation()
