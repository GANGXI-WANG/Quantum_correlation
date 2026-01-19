import numpy as np
import scipy.linalg

class SpinBosonSystem:
    def __init__(self, chain, omega_probe, Omega, Delta=0.0):
        """
        Initialize the system with a given ion chain and laser parameters.

        Args:
            chain (LinearChain): The ion chain object.
            omega_probe (float): Laser detuning (frequency difference) from carrier,
                                 but actually 'mu' in Hamiltonian is usually detuning.
                                 Here omega_probe is the 'mu' we inspect.
                                 Wait, user says "Delta denotes the laser detuning in the rotating frame".
                                 And J(omega) is probed at 'omega'.
            Omega (float): Rabi frequency.
            Delta (float): Detuning for the spin Hamiltonian term.
        """
        self.chain = chain
        self.omega_probe = omega_probe # This is for J_ij evaluation usually?
        self.Omega = Omega
        self.Delta = Delta

        # Get modes
        # We assume they are already computed and scaled
        if chain.frequencies is None:
            raise ValueError("Chain modes not computed.")

        self.mode_freqs = chain.frequencies
        self.mode_vecs = chain.eigenvectors

        # Lamb-Dicke parameters
        # eta_k = delta k * sqrt(hbar / 2 m w_k).
        # We need physical units or assume simplified model.
        # User says: "lambda_k = 2 eta_k b_k Omega".
        # We need eta_k.
        # Typically eta ~ 0.1.
        # It depends on mode frequency: eta_k = eta_0 / sqrt(w_k/w_0)?
        # Or constant?
        # "eta_k is the Lamb-Dicke parameter".
        # Usually eta_k = k_laser * sqrt(hbar/2 M w_k). So proportional to 1/sqrt(w_k).
        # Let's assume eta * sqrt(w_com / w_k).
        self.eta_0 = 0.1 # Typical value
        self.eta_k = self.eta_0 * np.sqrt(self.mode_freqs[-1] / self.mode_freqs) # scaled by COM freq (max)

    def calculate_coupling_matrix(self, omega):
        """
        Calculate J_ij(omega).

        J_ij(omega) = Sum_k ( |2 eta_k Omega|^3 |b_ik b_jk|^1.5 / sqrt(2) ) / ( (omega - w_k)^2 + 2(eta_k Omega)^2 |b_ik b_jk| )

        Note: The user formula has broadening Gamma_k = 2(eta_k Omega)^2 |b_ik b_jk| ?
        User eq:
        J_ij = sum frac{ |2 eta_k Omega|^3 |b_ik b_jk|^1.5 / sqrt(2) } { (w - w_k)^2 + 2(eta_k Omega)^2 |b_ik b_jk| }

        This looks like a Lorentzian with width gamma = sqrt(2) * ...?
        Actually, let's match the user formula exactly.

        Numerator: N_k = |2 eta_k Omega|^3 * |b_ik * b_jk|**(1.5) / sqrt(2)
        Denominator: D_k = (omega - w_k)**2 + 2 * (eta_k * Omega)**2 * |b_ik * b_jk|

        Wait, usually denominator is (w-wk)^2 + (gamma/2)^2 or something.
        Here the term added is 2(eta Omega)^2 |b b|.
        """
        N = self.chain.N
        J = np.zeros((N, N))

        # Precompute common terms
        # factor = 2 * eta_k * Omega
        factor = 2 * self.eta_k * self.Omega
        factor_cubed = np.abs(factor)**3

        # Denominator linewidth term: 2(eta Omega)^2 |b b|
        # This is 0.5 * (2 eta Omega)^2 = 0.5 * factor^2
        broadening_factor = 2 * (self.eta_k * self.Omega)**2

        b = self.mode_vecs # Shape (N_ions, N_modes)
        w_k = self.mode_freqs

        # Loop over i, j
        # Vectorized over k
        for i in range(N):
            for j in range(N):
                prod_b = b[i, :] * b[j, :]
                abs_prod_b = np.abs(prod_b)

                numerator = (factor_cubed * abs_prod_b**(1.5)) / np.sqrt(2)
                denominator = (omega - w_k)**2 + broadening_factor * abs_prod_b

                J[i, j] = np.sum(numerator / denominator)

        return J

    def construct_hamiltonian(self, ion_indices):
        """
        Construct Hamiltonian for a subset of ions (e.g., 2 ions).
        H = sum w_k a_k^d a_k + Delta/2 sum sigma_z + Omega sum sum eta_k b_ik (sigma+ a + sigma- a^d)

        We work in a truncated basis.
        Basis: |s1, s2, n1, n2, ..., nN>
        Since N=20 modes, we can't do full Fock space.
        But we only need single excitation subspace if we start with 1 excitation?
        The user wants "Microscopic dynamics... Time evolution yields pure state".
        And "Entanglement of Formation". This usually requires 2 qubits.
        If we have 2 qubits, states like |gg>, |ge>, |eg>, |ee>.
        And phonon bath.

        If we assume the "spin-boson" model is in the RWA (excitation conserving),
        H commutes with N_exc = sum |e><e| + sum a^d a.

        If we start in |e, g, 0> (1 excitation), we stay in 1 excitation subspace.
        Dim = 2 (spins) + N (phonons). Very small.
        However, entanglement usually requires superposition of |eg> and |ge>, which are in the same subspace.
        But to get entanglement starting from separable, we need interaction.
        Effective interaction is |eg> <-> |gg, 1_k> <-> |ge>.
        So |e,g,0> evolves to c1|e,g,0> + c2|g,e,0> + sum ck |g,g, 1_k>.
        If we trace out phonons, we get rho_spin in {|eg>, |ge>, |gg>}.
        Wait, what about |ee>?
        If we start with |e,e,0> (2 excitations), we couple to |e,g, 1_k> and |g,e, 1_k>, and |g,g, 2_k> or |g,g, 1_k, 1_q>.

        2-excitation subspace dimension:
        Spins: |ee> (1 state)
        Spins: |eg>, |ge> (2 states) * N modes (1 phonon) -> 2N states
        Spins: |gg> * (N + N(N-1)/2) modes (2 phonons) -> N(N+1)/2 states.
        Total ~ 1 + 40 + 210 ~ 250.
        This is small enough for exact diagonalization!

        So we will implement the Hamiltonian in the 0, 1, and 2 excitation subspaces.
        Or just 1 excitation if the user only cares about single particle transfer?
        "information exchange... decoherence...".
        "Entanglement of formation".
        To have entanglement, we need a mixed state of 2 qubits?
        If we start with |psi(0)> = |e, g, 0ph>, the state stays pure.
        Reduced state rho_AB is mixed.
        Entanglement of Formation for 2 qubits is defined for mixed states.

        Most experiments start with e.g. |up, down>.
        I will implement the solver for up to 2 excitations to be safe and general.
        Actually, for 2 ions, N_exc is conserved.
        If we start with |up, down>, we have 1 excitation.
        Then we only explore |up, down, 0>, |down, up, 0>, and |down, down, 1_k>.
        We never reach |up, up> or |down, down, 2_k>.

        Wait, does H conserve excitation?
        H = ... + (sigma+ a + sigma- a^d).
        Yes, sigma+ adds spin exc, removes phonon.
        So Total Exc = Spin Exc + Phonon Exc is conserved.

        So if we initialize in |e, g> (1 exc), we only need the 1-excitation subspace.
        Dimension = 2 + 20 = 22.

        But if we initialize in |e, e> (2 exc), we need 2-exc subspace.
        If we initialize in |g, g> (0 exc), trivial.

        The user asks for "entanglement of formation".
        If we start with |e, g>, we can get entangled |e,g> + |g,e>.
        This is Bell-like.

        I will implement the 1-excitation subspace solver first, as it's standard for |up, down> evolution.
        And allow for 2-exc if needed (but 1 is sufficient for |eg> -> entangled).

        Indices:
        Spins: 0, 1 (subset indices).
        Basis states:
        0: |e, g, 0>
        1: |g, e, 0>
        2...N+1: |g, g, 1_k>  (where k=0..N-1)

        Wait, for 2 ions, 'e' means sigma_z=+1?
        Hamiltonian: Delta/2 sigma_z.
        Usually |e> is +1, |g> is -1.

        Matrix construction:
        Basis size M = 2 + N_modes.

        H_mat elements:

        Diagonal:
        <e,g,0|H|e,g,0> = Delta/2 (1) + Delta/2 (-1) = 0.
        <g,e,0|H|g,e,0> = 0.
        <g,g,1_k|H|g,g,1_k> = w_k + Delta/2(-1) + Delta/2(-1) = w_k - Delta.

        Off-diagonal:
        Coupling: Omega * eta_k * b_{i,k} (sigma_i^+ a_k + h.c.)

        <e,g,0| H |g,g,1_k>
        Term: sigma_1^+ a_k connects |g,g,1_k> to |e,g,0>.
        Coeff: Omega * eta_k * b_{ion1, k}.
        So H_{0, 2+k} = Omega * eta_k * b_{ion1, k}.

        <g,e,0| H |g,g,1_k>
        Term: sigma_2^+ a_k connects |g,g,1_k> to |g,e,0>.
        Coeff: Omega * eta_k * b_{ion2, k}.
        So H_{1, 2+k} = Omega * eta_k * b_{ion2, k}.

        Returns: H matrix, Basis info.
        """
        N_modes = self.chain.N
        dim = 2 + N_modes
        H = np.zeros((dim, dim), dtype=complex)

        ion1 = ion_indices[0]
        ion2 = ion_indices[1]

        # Diagonal elements
        # States 0, 1 have 0 phonons, and (up, down) or (down, up).
        # Energy = Delta/2 - Delta/2 = 0.

        # States 2..2+N have 1 phonon k, and (down, down).
        # Energy = w_k - Delta/2 - Delta/2 = w_k - Delta.

        for k in range(N_modes):
            H[2+k, 2+k] = self.mode_freqs[k] - self.Delta

        # Off-diagonal
        # Coupling constant g_{i,k} = Omega * eta_k * b_{i,k}
        # Note: eta_k needs to be correct.

        g1 = self.Omega * self.eta_k * self.mode_vecs[ion1, :]
        g2 = self.Omega * self.eta_k * self.mode_vecs[ion2, :]

        for k in range(N_modes):
            # Coupling |e,g,0> <-> |g,g,1_k>
            H[0, 2+k] = g1[k]
            H[2+k, 0] = np.conj(g1[k])

            # Coupling |g,e,0> <-> |g,g,1_k>
            H[1, 2+k] = g2[k]
            H[2+k, 1] = np.conj(g2[k])

        return H

    def simulate_dynamics(self, ion_indices, time_points, initial_state_type='eg'):
        """
        Simulate evolution.
        initial_state_type:
            'eg' -> |e, g, 0> (index 0)
            'bell' -> (|e, g, 0> + |g, e, 0>) / sqrt(2) (superposition of index 0 and 1)
        """
        H = self.construct_hamiltonian(ion_indices)

        psi0 = np.zeros(H.shape[0], dtype=complex)

        if initial_state_type == 'eg':
            psi0[0] = 1.0
        elif initial_state_type == 'bell':
            psi0[0] = 1.0/np.sqrt(2)
            psi0[1] = 1.0/np.sqrt(2)
        else:
            raise ValueError("Unknown initial state type. Use 'eg' or 'bell'.")

        # Evolve
        # psi(t) = exp(-iHt) psi0
        # Since H is small constant matrix, we can diagonalize it.
        evals, evecs = np.linalg.eigh(H)

        # c_n = <n|psi0>
        coeffs = np.dot(evecs.conj().T, psi0)

        # psi(t) = sum c_n e^{-i E_n t} |n>

        # We want density matrix of spins.
        # States: 0->|eg>, 1->|ge>, 2..->|gg>
        # rho_spin basis: |eg>, |ge>, |gg>, |ee> (ee is zero)
        # Actually, let's output the full rho_spin(t) in basis |eg>, |ge>, |gg>.

        dim_s = 3
        results = []

        for t in time_points:
            psi_t_coeffs = coeffs * np.exp(-1j * evals * t)
            psi_t = np.dot(evecs, psi_t_coeffs)

            # Trace out phonons
            # Basis: 0 (|eg, 0>), 1 (|ge, 0>), 2..N+1 (|gg, k>)

            c_eg = psi_t[0]
            c_ge = psi_t[1]
            c_gg_k = psi_t[2:]

            # rho_spin elements
            # |eg><eg|: |c_eg|^2
            # |ge><ge|: |c_ge|^2
            # |eg><ge|: c_eg * c_ge*
            # |gg><gg|: sum_k |c_gg_k|^2
            # Cross terms between |eg/ge> and |gg> are zero because they have orthogonal phonon parts (0 vs 1_k)

            rho = np.zeros((4, 4), dtype=complex)
            # Basis: ee, eg, ge, gg
            # But we have no ee.

            rho[1, 1] = np.abs(c_eg)**2      # eg, eg
            rho[2, 2] = np.abs(c_ge)**2      # ge, ge
            rho[1, 2] = c_eg * np.conj(c_ge) # eg, ge
            rho[2, 1] = np.conj(rho[1, 2])
            rho[3, 3] = np.sum(np.abs(c_gg_k)**2) # gg, gg

            results.append(rho)

        return np.array(results)
