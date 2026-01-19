import numpy as np
from scipy.optimize import minimize

class LinearChain:
    def __init__(self, N=20, alpha=0.0):
        """
        Initialize the linear ion chain.

        Args:
            N (int): Number of ions.
            alpha (float): Strength of the quartic term in the axial potential relative to quadratic.
                           Potential V(z) ~ 0.5*z^2 + alpha*z^4.
                           Dimensionless units.
        """
        self.N = N
        self.alpha = alpha
        self.positions = None
        self.modes = None
        self.frequencies = None
        self.eigenvectors = None
        self.raw_evals = None
        self.raw_evecs = None

    @classmethod
    def from_data(cls, dz, omegas):
        """
        Initialize chain from inter-ion distances and mode frequencies.

        Args:
            dz (list or np.array): Distances between adjacent ions (length N-1).
            omegas (list or np.array): Mode frequencies (length N).
        """
        N = len(dz) + 1
        if len(omegas) != N:
            raise ValueError(f"Number of frequencies ({len(omegas)}) must equal number of ions ({N}) implied by dz.")

        instance = cls(N=N)

        # Calculate positions centered at 0
        z = np.zeros(N)
        # z[0] is arbitrary, fix later to center
        current_z = 0.0
        z[0] = 0.0
        for i, d in enumerate(dz):
            current_z += d
            z[i+1] = current_z

        # Center
        center = (z[0] + z[-1]) / 2.0
        instance.positions = z - center

        # Store frequencies
        # User provided omegas might be sorted high to low.
        instance.frequencies = np.array(omegas)

        return instance

    def potential(self, u):
        """
        Dimensionless potential energy of the chain.
        u: array of positions.
        """
        # Sort positions to avoid singularities in calculation (though minimize should handle order if started correctly)
        # But we assume ordered for the sum 1/|ui-uj|
        # To avoid sorting in the loop, we assume u is ordered, or we take abs.

        # Confinement
        V_trap = 0.5 * np.sum(u**2) + self.alpha * np.sum(u**4)

        # Coulomb
        # Vectorized coulomb sum
        u_col = u[:, np.newaxis]
        diff = u_col - u
        # We need upper triangle
        with np.errstate(divide='ignore'):
            inv_dist = 1.0 / np.abs(diff)

        # Remove diagonal (infinity)
        inv_dist[np.isinf(inv_dist)] = 0

        V_coulomb = 0.5 * np.sum(inv_dist) # 0.5 because we summed both i,j and j,i

        return V_trap + V_coulomb

    def force(self, u):
        """
        Gradient of potential (negative force).
        """
        N = self.N
        grad = u + 4 * self.alpha * u**3 # Trap force

        u_col = u[:, np.newaxis]
        diff = u_col - u # i - j

        # Coulomb force on i from j: F_ij = sgn(z_i - z_j) / |z_i - z_j|^2
        # dV/du_i = - sum_j F_ij
        # But here we calculate gradient of V directly.
        # V = sum 1/|ui - uj|
        # d(1/|x|)/dx = -sgn(x)/x^2 = -x/|x|^3

        with np.errstate(divide='ignore', invalid='ignore'):
            forces = - diff / np.abs(diff)**3

        forces[np.isnan(forces)] = 0

        grad += np.sum(forces, axis=1)
        return grad

    def compute_equilibrium_positions(self):
        """
        Find equilibrium positions minimizing the potential.
        """
        # Initial guess: approximate length scale
        # For harmonic, L ~ (3N log N)^(1/3).
        # Just create a linear array spread out
        L_approx = 2.0 * self.N**(1/3) # Rough guess
        initial_guess = np.linspace(-L_approx, L_approx, self.N)

        res = minimize(self.potential, initial_guess, jac=self.force, method='L-BFGS-B',
                       options={'ftol': 1e-12, 'gtol': 1e-12})

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self.positions = np.sort(res.x)
        return self.positions

    def compute_transverse_modes(self):
        """
        Calculate transverse normal modes.
        Returns eigenvalues (lambda) and eigenvectors (b).
        The actual frequencies will be sqrt(w_trans^2 - w_axial^2 * lambda).
        Here we return lambda (the Coulomb matrix eigenvalues).
        """
        if self.positions is None:
            self.compute_equilibrium_positions()

        u = self.positions
        N = self.N

        # Transverse Hessian A_ij
        # Diagonal: A_ii = sum_{j!=i} 1/|u_i - u_j|^3
        # Off-diagonal: A_ij = -1/|u_i - u_j|^3

        u_col = u[:, np.newaxis]
        diff = u_col - u

        with np.errstate(divide='ignore'):
            term = 1.0 / np.abs(diff)**3
        term[np.isinf(term)] = 0

        A = -term

        # Diagonal elements
        row_sum = np.sum(term, axis=1)
        np.fill_diagonal(A, row_sum)

        # Diagonalize
        evals, evecs = np.linalg.eigh(A)

        # Sort by eigenvalue? Usually they come sorted.
        # evecs columns are eigenvectors.

        self.raw_evals = evals
        self.raw_evecs = evecs

        # If we have pre-assigned frequencies (via from_data), we need to pair them.
        if self.frequencies is not None:
             self._pair_modes_with_frequencies()

        return evals, evecs

    def _pair_modes_with_frequencies(self):
        """
        Pair the calculated eigenvectors with the stored frequencies.
        Transverse physics: Smallest eigenvalue (stiffest repulsion diff) -> Highest Frequency.
        Actually, Transverse freq^2 = w_trap^2 - const * eigenvalue.
        So Small eigenvalue -> Large frequency.
        """
        if self.frequencies is None or self.raw_evals is None:
            return

        # Sort eigenvectors by eigenvalue (ascending)
        # raw_evals and raw_evecs are already sorted ascending by eigh.

        # Sort provided frequencies descending (Highest freq corresponds to Lowest eigenvalue)
        sorted_freqs = np.sort(self.frequencies)[::-1]

        self.frequencies = sorted_freqs
        # self.raw_evecs corresponds to eigenvalues Ascending.
        # so raw_evecs[:, 0] corresponds to Lowest eigenvalue -> Highest Frequency.

        self.eigenvectors = self.raw_evecs

    def get_scaled_modes(self, f_min, f_max):
        """
        Scale the modes to fit the provided frequency range [f_min, f_max].
        We assume the relationship: omega_k^2 = C - D * lambda_k
        Or omega_k^2 = omega_trap^2 - lambda_k * omega_z^2.

        However, to exactly match the min and max frequencies provided by the user,
        we can simply map the range of eigenvalues to the range of squared frequencies.

        Transverse modes: Higher lambda (stiffest Coulomb repulsion difference) -> Lower frequency?
        Wait.
        Equation of motion: d^2x/dt^2 = - (omega_t^2 - sum 1/d^3) x ...
        Actually, the Coulomb interaction softens the trap in transverse direction.
        So larger Coulomb curvature (positive sum 1/d^3) -> Lower frequency.
        So omega_k^2 = omega_common^2 - lambda_k.

        Where lambda_k are eigenvalues of the A matrix calculated above.
        Min lambda -> Max omega.
        Max lambda -> Min omega.

        We have f_min and f_max.
        We have lambda_min and lambda_max.

        omega_max^2 = C - lambda_min
        omega_min^2 = C - lambda_max

        We can solve for C.
        """
        if self.raw_evals is None:
            self.compute_transverse_modes()

        evals = self.raw_evals # lambda

        lambda_min = np.min(evals)
        lambda_max = np.max(evals)

        w_min = 2 * np.pi * f_min
        w_max = 2 * np.pi * f_max

        # w_max corresponds to lambda_min
        # w_min corresponds to lambda_max
        # w^2 = Offset - Slope * lambda

        # w_max^2 = Off - S * l_min
        # w_min^2 = Off - S * l_max
        # (w_max^2 - w_min^2) = S * (l_max - l_min)

        slope = (w_max**2 - w_min**2) / (lambda_max - lambda_min)
        offset = w_max**2 + slope * lambda_min

        squared_freqs = offset - slope * evals

        # Check for negative squared freqs (instability)
        if np.any(squared_freqs < 0):
            # This shouldn't happen if we map directly, unless the range is flipped.
            raise ValueError("Scaling resulted in imaginary frequencies.")

        freqs = np.sqrt(squared_freqs)

        # Sort frequencies and eigenvectors (typically we want frequencies low to high?)
        # But the user might expect them ordered by mode index.
        # The eigenvalues 'evals' are sorted low to high.
        # So 'freqs' will be sorted high to low (because of minus sign).
        # Let's sort them low to high to be standard.

        sort_idx = np.argsort(freqs)
        self.frequencies = freqs[sort_idx]
        self.eigenvectors = self.raw_evecs[:, sort_idx]

        return self.frequencies, self.eigenvectors
