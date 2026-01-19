import numpy as np
import matplotlib.pyplot as plt
from src.ion_chain import LinearChain

def test_chain_reproduction():
    print("Testing LinearChain reproduction of external logic...")

    dz = [6.0019812,5.15572782,4.72673127,4.54681994,4.47707492,4.54681994,4.72673127,5.15572782,6.0019812]
    omegas = 2*np.pi*np.array([242412, 242401, 242389, 242374, 242356, 242335, 242311, 242284, 242255, 242223]) # rad/s

    # In external code:
    # omx = max(omega) - 2.0 * np.pi * 240.0203e3
    # Wait, 240.0203e3 is 240 kHz.
    # The omegas are around 242 kHz.
    # So omx (trap freq?) is small? ~ 2 kHz?
    # No, that subtraction is `omega_k = omega - shift`.
    # But `omx` variable in external code is:
    # omx = max(omega) - 2.0 * np.pi * 240.0203e3
    # Let's verify what `omx` is passed to `collective_mode`.

    shift = 2.0 * np.pi * 240.0203e3
    omx_external = np.max(omegas) - shift

    print(f"External omx (rad/s): {omx_external}")
    print(f"External omx (kHz): {omx_external / (2*np.pi) / 1e3}")

    # This `omx` seems extremely small (2.4 kHz) compared to 242 kHz.
    # Maybe the frequencies provided in `omega` are absolute, but the calculation is done in a frame?
    # Or `omx` is the *detuning*?
    # `V[i,i] = ... + omx**2`.
    # If omx is small, the trap is weak.

    chain = LinearChain.from_data(dz, omegas)
    raw_freqs, evecs = chain.compute_transverse_modes(omx_external)

    # Get sorted High to Low
    calc_freqs, calc_modes = chain.get_modes()

    print("Calculated Top 3 Frequencies (Hz):", calc_freqs[:3] / (2*np.pi))
    print("Provided Top 3 Frequencies (Hz):", omegas[:3] / (2*np.pi))

    # Check if they match
    # Note: `ion_distance_parameter.py` returns `omega_k - max`.
    # It prints `omega_k` (from data) and then `omega_k_cal` (relative).
    # We want to see if our `calc_freqs` match `np.sqrt(E)`.

    # To check consistency:
    # The external code defines `omega` array as the desired result?
    # And it uses `dz` and `omx` to calculate modes.
    # Does the calculated mode match the `omega` array (shifted)?
    # Or is `omega` array just experimental data and we simulate with calculated ones?
    # Usually we use calculated ones for consistency with eigenvectors.

    print("Calculated Freq Range (Hz):", calc_freqs[-1]/(2*np.pi), "to", calc_freqs[0]/(2*np.pi))

    # Check eigenvectors
    # Ion 0, mode 0 (Highest freq)
    b_00 = calc_modes[0, 0]
    print(f"Eigenvector component (0,0): {b_00}")

    # Check COM
    if np.all(calc_modes[:, 0] > 0) or np.all(calc_modes[:, 0] < 0):
        print("Mode 0 is COM-like.")
    else:
        print("Mode 0 is NOT COM-like.")

if __name__ == "__main__":
    test_chain_reproduction()
