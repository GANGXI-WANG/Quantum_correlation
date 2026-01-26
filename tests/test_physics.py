import numpy as np
import matplotlib.pyplot as plt
from src.ion_chain import LinearChain
from src.physics import SpinBosonSystem

def test_physics_module():
    print("Testing Physics Module with 10-ion parameters...")

    # Setup 10-ion chain
    dz = [6.0019812,5.15572782,4.72673127,4.54681994,4.47707492,4.54681994,4.72673127,5.15572782,6.0019812]
    omegas = 2*np.pi*np.array([242412, 242401, 242389, 242374, 242356, 242335, 242311, 242284, 242255, 242223])
    chain = LinearChain.from_data(dz, omegas)

    # Calculate modes with omx
    shift = 2.0 * np.pi * 240.0203e3
    omx_external = np.max(omegas) - shift
    chain.compute_transverse_modes(omx_external)

    # Manually align eigenvectors if needed (in main.py we do it, here we assume chain has them)
    # But compute_transverse_modes sets raw_freqs and eigenvectors.
    # It does NOT align them to provided 'omegas'.
    # get_modes() returns sorted High->Low.
    _, evecs = chain.get_modes()
    chain.eigenvectors = evecs

    # Parameters
    Omega = 2 * np.pi * 400e3
    eta = 0.06
    omega_probe = omegas[0]

    sys = SpinBosonSystem(chain, omega_probe, Omega)
    # Manually set eta (checking if class allows)
    sys.eta_k = np.full(chain.N, eta)

    # Test Coupling Matrix
    print("Calculating J_ij...")
    J = sys.calculate_coupling_matrix(omega_probe)
    print(f"J matrix shape: {J.shape}")
    assert J.shape == (10, 10)

    # Check symmetry
    if np.allclose(J, J.T):
        print("J matrix is symmetric.")
    else:
        print("Error: J matrix is not symmetric.")

    # Check values range
    print(f"J values range: {np.min(J):.2e} to {np.max(J):.2e}")
    # With large Omega (400kHz) and eta (0.06), J should be significant.
    # eta*Omega ~ 24 kHz.
    # J ~ (eta*Omega)^3 / gamma^2.

    # Test Hamiltonian
    print("Constructing Hamiltonian...")
    H = sys.construct_hamiltonian([4, 5]) # Middle ions
    print(f"H shape: {H.shape}")
    expected_dim = 2 + 10 # 12
    assert H.shape == (expected_dim, expected_dim)

    # Test Dynamics
    print("Simulating Dynamics...")
    t = np.linspace(0, 10e-6, 10)
    rhos = sys.simulate_dynamics([4, 5], t)
    print(f"Rhos shape: {rhos.shape}")

    # Check trace = 1
    traces = np.trace(rhos, axis1=1, axis2=2)
    print(f"Trace deviation: {np.max(np.abs(traces - 1.0)):.2e}")
    assert np.allclose(traces, 1.0)

    print("Physics module tests passed.")

if __name__ == "__main__":
    test_physics_module()
