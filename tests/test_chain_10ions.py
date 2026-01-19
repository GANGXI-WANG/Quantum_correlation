import numpy as np
import matplotlib.pyplot as plt
from src.ion_chain import LinearChain

def test_chain_10ions():
    print("Testing LinearChain with 10-ion data...")

    dz = [6.0019812,5.15572782,4.72673127,4.54681994,4.47707492,4.54681994,4.72673127,5.15572782,6.0019812]
    omegas = 2*np.pi*np.array([242412, 242401, 242389, 242374, 242356, 242335, 242311, 242284, 242255, 242223])

    chain = LinearChain.from_data(dz, omegas)

    # Check positions
    print(f"Number of ions: {chain.N}")
    print(f"Positions range: {chain.positions[0]:.2f} to {chain.positions[-1]:.2f}")

    # Verify symmetry
    if np.allclose(chain.positions, -chain.positions[::-1], atol=1e-5):
        print("Positions are symmetric.")
    else:
        print("Warning: Positions are not symmetric.")

    # Compute modes
    evals, evecs = chain.compute_transverse_modes()

    # Check if frequencies match provided
    # The class should have sorted them descending
    print("Top 3 stored frequencies (Hz):", chain.frequencies[:3] / (2*np.pi))
    print("Top 3 provided frequencies (Hz):", omegas[:3] / (2*np.pi))

    # Check alignment: First mode (Highest Freq) should be COM-like (eigenvector elements same sign)
    com_mode = chain.eigenvectors[:, 0] # Highest freq
    if np.all(com_mode > 0) or np.all(com_mode < 0):
        print("Highest frequency mode is COM-like (all same sign). Correct.")
    else:
        print("Warning: Highest frequency mode is NOT COM-like.")
        print("Eigenvector:", com_mode)

    # Check Second mode (Tilt)
    tilt_mode = chain.eigenvectors[:, 1]
    # Should be antisymmetric
    if np.allclose(tilt_mode, -tilt_mode[::-1], atol=1e-2):
        print("Second highest mode is antisymmetric (Tilt-like). Correct.")
    else:
        print("Warning: Second mode is not antisymmetric.")

    print("10-ion chain test passed.")

if __name__ == "__main__":
    test_chain_10ions()
