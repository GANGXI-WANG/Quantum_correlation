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
    omegas = 2*np.pi*np.array([242412, 242401, 242389, 242374, 242356, 242335, 242311, 242284, 242255, 242223])

    Omega = 2 * np.pi * 400e3 # 400 kHz in rad/s
    eta = 0.06 # Constant eta

    print("Initializing 10-Ion Chain...")
    chain = LinearChain.from_data(dz, omegas)

    # Calculate modes using the specific omx from external code logic
    # omx = max(omega) - shift
    shift = 2.0 * np.pi * 240.0203e3
    omx_external = np.max(omegas) - shift

    # Compute modes with this omx to get the correct b_jk
    # The frequencies returned will be the calculated ones (small), but we will overwrite them
    # with the provided 'omegas' (large) as per LinearChain.from_data logic which stores them.
    # Actually from_data stores 'omegas' in chain.frequencies.
    # compute_transverse_modes overwrites them?
    # No, my updated compute_transverse_modes calculates raw_freqs but doesn't overwrite self.frequencies unless we ask it to.
    # Wait, let's check LinearChain again.
    # It stores raw_freqs.
    # But SpinBosonSystem uses chain.frequencies.
    # We should ensure chain.frequencies is the provided 'omegas', but chain.eigenvectors is the calculated one.

    chain.compute_transverse_modes(omx_external)
    # LinearChain.get_modes() returns calculated freqs.
    # We want to keep the PROVIDED frequencies.
    # But we want the CALCULATED eigenvectors.
    # And we need to ensure they are sorted consistently (High to Low).
    # My LinearChain.get_modes returns High to Low.
    # And the provided omegas are High to Low.
    # So we just take the eigenvectors from get_modes and assign them to the chain.

    _, evecs = chain.get_modes()
    chain.eigenvectors = evecs
    # chain.frequencies is already set by from_data

    # Select probe frequency for heatmap
    w_center = np.mean(omegas)

    print("Generating J_ij Heatmap...")
    sys = SpinBosonSystem(chain, w_center, Omega)
    # Manually overwrite eta_k to be constant 0.06 as per user request
    sys.eta_k = np.full(chain.N, eta)

    J = sys.calculate_coupling_matrix(w_center)

    # Plot J_ij
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
    # Select two ions: 4 and 5 (indices, 0-based) - The middle pair
    ions = [4, 5]

    # Set Delta resonant with a mode? Or detuned?
    # Usually we want resonance to see exchange.
    # Let's set Delta to the COM mode frequency (highest)
    sys.Delta = omegas[0]

    t_max = 50e-6 # 50 us
    t_points = np.linspace(0, t_max, 100)

    rhos = sys.simulate_dynamics(ions, t_points)

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

    axes[0].plot(t_points*1e6, C12_vals, color='b')
    axes[0].set_ylabel('C_12')
    axes[0].set_title('Connected Correlator (Ions 4 & 5)')

    axes[1].plot(t_points*1e6, EoF_vals, color='g')
    axes[1].set_ylabel('Entanglement of Formation')

    axes[2].plot(t_points*1e6, QD_vals, color='r')
    axes[2].set_ylabel('Quantum Discord')
    axes[2].set_xlabel('Time (us)')

    plt.tight_layout()
    plt.savefig('Dynamics_10ions.png')
    print("Saved Dynamics_10ions.png")

if __name__ == "__main__":
    run_simulation()
