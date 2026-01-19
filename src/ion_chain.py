import numpy as np
from scipy.optimize import minimize

class LinearChain:
    def __init__(self, N=10):
        """
        Initialize the linear ion chain.
        N: Number of ions.
        """
        self.N = N
        self.positions = None
        self.frequencies = None
        self.eigenvectors = None

    @classmethod
    def from_data(cls, dz, omegas):
        """
        Initialize chain from inter-ion distances and mode frequencies.

        Args:
            dz (list or np.array): Distances between adjacent ions (length N-1).
                                   Units: micrometers (based on external code).
            omegas (list or np.array): Mode frequencies (length N).
        """
        N = len(dz) + 1
        instance = cls(N=N)

        # Calculate positions
        # External code: z[i+1] = z[i] + dz[i], starting z[0]=0.
        z = np.zeros(N)
        current_z = 0.0
        for i, d in enumerate(dz):
            current_z += d
            z[i+1] = current_z

        # Center the chain (optional but good practice, external code doesn't seem to center explicitly in collective_mode but relative distances matter)
        # External code uses rinv3 which depends on |z_i - z_j|, so absolute position doesn't matter.
        instance.positions = z

        # Frequencies provided are usually the target frequencies or measured ones.
        instance.frequencies = np.array(omegas)

        return instance

    def compute_transverse_modes(self, omx):
        """
        Calculate collective modes using the method from ion_distance_parameter.py

        Args:
            omx: The trap frequency (highest frequency mode usually).
                 External code uses omx as the 'trap frequency' in the diagonal term.
        """
        z = self.positions
        L = self.N

        # Distance matrix
        # rinv3_ij = 1 / |z_i - z_j|^3
        rinv3 = np.abs(np.reshape(z, (1, -1)) - np.reshape(z, (-1, 1)))
        np.fill_diagonal(rinv3, 1.0) # Avoid div by zero temporarily
        rinv3 = 1.0 / rinv3**3
        np.fill_diagonal(rinv3, 0.0)

        # Coefficient from external code
        # coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / 1e3**2 (if units are kHz?)
        # External code comment says: # 12.24 10ions
        # coef = ... / 1e-6**3 / 1e3**2 ?
        # The frequencies in external code 'omega' are ~242000.
        # If these are Hz, they are very low (kHz range).
        # Wait, usually trap freq is MHz. 242000 Hz = 242 kHz.
        # 171 Yb mass = 171 * 1.66e-27 kg.
        # Charge e.
        # Distances dz ~ 4-6. Units? usually microns.

        # Let's copy the coefficient calculation exactly.
        # coef = 9e9 * 1.602e-19**2 / (171 * 1.66e-27) / (1e-6)**3 / (1e3)**2
        # (1e3)**2 suggests scaling from Hz to kHz? Or similar.
        # If omx is in rad/s, we need consistent units.
        # External code `omx` seems to be in rad/s if it comes from `2*pi*array`.
        # BUT the code calculates `V = ... + omx**2`.
        # If `omx` is 2*pi*242412 ~ 1.5e6. Squared ~ 2e12.
        # Let's check `coef`.
        # 9e9 * (1.6e-19)^2 ~ 2.3e-28.
        # Mass ~ 2.8e-25.
        # Ratio ~ 1e-3.
        # 1/r^3 ~ 1/(1e-6)^3 = 1e18.
        # Total ~ 1e15.
        # If we divide by (1e3)^2 = 1e6, we get 1e9.
        # This matches omx^2 ~ 1e12 if omx is in kHz?

        # Wait, external code `omega` is `2*np.pi*np.array([242412...])`.
        # So omega is in rad/s (approx 1.5e6).
        # `omx` is `max(omega) - shift`.

        # Re-eval coefficient lines:
        # coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / (1e6)**2 (The deepseek file has (1e6)**2)
        # The ion_distance_parameter.py has `1e3**2`.
        # This difference (1e6 vs 1e3) squared is 1e6.
        # 1e3^2 = 1e6. 1e6^2 = 1e12.
        # If omx is ~ 1e6, omx^2 ~ 1e12.
        # If coef ~ 1e15 (raw SI)
        # If we divide by 1e6 (from 1e3^2), we get 1e9. Too small for 1e12?
        # If we divide by 1e12 (from 1e6^2), we get 1e3. Too small.

        # Let's trust the logic:
        # V matrix is calculated.
        # E, b_jk = eigh(V).
        # omega_k = sqrt(E).

        # I will implement the exact lines from `ion_distance_parameter.py` because that's the source of truth for the 10-ion parameters.
        # Note: `ion_distance_parameter.py` uses `1e3**2`.

        coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / 1e3**2

        V = coef * rinv3
        # Diagonal elements
        V[range(L), range(L)] = -np.sum(V, axis=1) + omx**2

        # Diagonalize
        evals, evecs = np.linalg.eigh(V)

        # Sort? eigh returns sorted eigenvalues.
        # External code returns `omega_k = np.sqrt(E)`.
        # And `omega_k - np.max(omega_k)`.

        # In `ion_distance_parameter.py`:
        # omega_k = np.sqrt(E)
        # return (omega_k - np.max(omega_k), b_jk)

        # The frequencies I want to store are the raw omega_k (sqrt(E)).

        self.raw_freqs = np.sqrt(evals)
        self.eigenvectors = evecs

        # We should ensure the sorted order matches the 'omegas' provided if we want to pair them.
        # Usually standard is Highest Freq = COM (for transverse?).
        # `ion_distance_parameter.py` calculates `omega_k`.
        # The `omega` array in that file is sorted high to low.
        # `omega_k` from `eigh` (sorted E low to high) implies frequencies low to high?
        # Wait, V = - Coulomb + Trap.
        # Smallest eigenvalue of V -> Smallest frequency?
        # Yes.
        # So `self.raw_freqs` are Low to High.
        # The provided `omegas` are High to Low.

        return self.raw_freqs, self.eigenvectors

    def get_modes(self):
        """
        Return frequencies and eigenvectors sorted High to Low to match provided data.
        """
        if self.eigenvectors is None:
            return None, None

        # Reverse order to match High->Low
        freqs = self.raw_freqs[::-1]
        modes = self.eigenvectors[:, ::-1]

        return freqs, modes
