import numpy as np
import matplotlib.pyplot as plt
from src.ion_chain import LinearChain
from src.physics import SpinBosonSystem

def test_physics_module():
    print("Testing Physics Module...")

    # Setup simple chain
    N = 20
    chain = LinearChain(N=N, alpha=0.001)
    chain.compute_transverse_modes()
    chain.get_scaled_modes(2.133e6, 2.406e6)

    omega_probe = 2.4e6
    Omega = 100e3 # 100 kHz

    sys = SpinBosonSystem(chain, omega_probe, Omega)

    # Test Coupling Matrix
    print("Calculating J_ij...")
    J = sys.calculate_coupling_matrix(omega_probe)
    print(f"J matrix shape: {J.shape}")
    assert J.shape == (N, N)

    # Check symmetry
    if np.allclose(J, J.T):
        print("J matrix is symmetric.")
    else:
        print("Error: J matrix is not symmetric.")

    # Check values range
    print(f"J values range: {np.min(J):.2e} to {np.max(J):.2e}")

    # Test Hamiltonian
    print("Constructing Hamiltonian...")
    H = sys.construct_hamiltonian([9, 10]) # Middle ions
    print(f"H shape: {H.shape}")
    expected_dim = 2 + N
    assert H.shape == (expected_dim, expected_dim)

    # Test Dynamics
    print("Simulating Dynamics...")
    t = np.linspace(0, 10e-6, 10)
    rhos = sys.simulate_dynamics([9, 10], t)
    print(f"Rhos shape: {rhos.shape}")

    # Check trace = 1
    traces = np.trace(rhos, axis1=1, axis2=2)
    print(f"Trace deviation: {np.max(np.abs(traces - 1.0)):.2e}")
    assert np.allclose(traces, 1.0)

    print("Physics module tests passed.")

if __name__ == "__main__":
    test_physics_module()
