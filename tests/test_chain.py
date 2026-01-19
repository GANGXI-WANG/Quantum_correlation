import numpy as np
import matplotlib.pyplot as plt
from src.ion_chain import LinearChain

def test_chain_spectrum():
    print("Testing LinearChain...")
    N = 20
    # Try with slight quartic to match description "weak quartic"
    # Though we don't know the value, let's try 0.001
    chain = LinearChain(N=N, alpha=0.001)

    pos = chain.compute_equilibrium_positions()
    print(f"Computed positions for {N} ions. Range: {pos[0]:.2f} to {pos[-1]:.2f}")

    # Check symmetry
    if not np.allclose(pos, -pos[::-1], atol=1e-5):
        print("Warning: Positions are not symmetric.")
    else:
        print("Positions are symmetric.")

    # Calculate modes
    evals, evecs = chain.compute_transverse_modes()
    print(f"Computed raw eigenvalues. Range: {np.min(evals):.4f} to {np.max(evals):.4f}")

    # Scale
    f_min = 2.133e6
    f_max = 2.406e6
    freqs, modes = chain.get_scaled_modes(f_min, f_max)

    print(f"Scaled Frequencies (MHz): {freqs/1e6}")

    # Verify range
    assert np.isclose(freqs[0], f_min * 2 * np.pi) or np.isclose(freqs[0], f_max * 2 * np.pi)
    assert np.isclose(freqs[-1], f_min * 2 * np.pi) or np.isclose(freqs[-1], f_max * 2 * np.pi)

    print(f"Min Freq: {np.min(freqs)/2/np.pi/1e6:.5f} MHz")
    print(f"Max Freq: {np.max(freqs)/2/np.pi/1e6:.5f} MHz")

    # Check density of states
    # Plot histogram
    plt.figure()
    plt.hist(freqs/2/np.pi/1e6, bins=10)
    plt.title("Mode Density")
    plt.savefig("tests/mode_density.png")
    print("Saved mode density plot.")

    # Check "dense near COM".
    # COM mode is usually the one with eigenvector [1, 1, ..., 1] (normalized).
    # Let's find which mode has max overlap with COM vector.
    com_vector = np.ones(N) / np.sqrt(N)
    overlaps = np.abs(np.dot(com_vector, modes))
    com_mode_idx = np.argmax(overlaps)

    print(f"COM mode index: {com_mode_idx} (overlap {overlaps[com_mode_idx]:.4f})")
    print(f"COM mode frequency: {freqs[com_mode_idx]/2/np.pi/1e6:.5f} MHz")

    # If the user says "dense near COM", we expect many modes near this frequency.
    # In transverse chain, COM is highest freq.
    if com_mode_idx == N-1: # Highest index
        print("COM mode is the highest frequency mode (Typical for Transverse).")
    elif com_mode_idx == 0:
        print("COM mode is the lowest frequency mode (Typical for Axial).")
    else:
        print("COM mode is in the middle.")

    # Density check
    # Check spacing near COM vs other end
    spacings = np.diff(freqs)
    print(f"Mean spacing: {np.mean(spacings)/2/np.pi/1e6:.4f} MHz")
    print(f"Spacing near top (COM): {spacings[-1]/2/np.pi/1e6:.4f} MHz")
    print(f"Spacing near bottom: {spacings[0]/2/np.pi/1e6:.4f} MHz")

    if spacings[-1] < spacings[0]:
        print("Spectrum is denser at high frequencies (COM end). Matches description.")
    else:
        print("Spectrum is denser at low frequencies. Might not match description.")

if __name__ == "__main__":
    test_chain_spectrum()
