import numpy as np
from scipy.sparse import diags, kron, identity, csr_matrix
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm, logm  # ADDITION: For correlation calculations
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os

class TwoSpinsInChain:
    """
    N 离子链（产生 N 个集体声子模），但只研究链上任意两个被寻址离子 (i, j) 的自旋-声子耦合动力学。
    [Original docstring remains]
    """
    def __init__(self, trap_config, addressed=(0,1), g0=0.0067, max_total_dim=1_000_000):
        # [Keep all original initialization code exactly as it was]
        self.num_ions = int(trap_config['num_ions'])
        self.trap_frequency_MHz = float(trap_config['trap_frequency'])
        self.trap_frequency = self.trap_frequency_MHz * 2 * np.pi
        self.g0_MHz = float(g0)
        self.g0 = self.g0_MHz * 2 * np.pi

        # 修正：正确调用_default_ion_positions方法
        if trap_config.get('ion_positions') is None:
            self.ion_positions = self._default_ion_positions()  # 调用方法而非直接赋值
        else:
            self.ion_positions = np.array(trap_config['ion_positions'], dtype=float)
            if self.ion_positions.size != self.num_ions:
                raise ValueError("ion_positions length mismatch with num_ions")

        self.addressed = tuple(addressed)
        if len(self.addressed) != 2:
            raise ValueError("addressed must be a tuple/list of two indices")
        i0, i1 = self.addressed
        if not (0 <= i0 < self.num_ions and 0 <= i1 < self.num_ions and i0 != i1):
            raise ValueError("addressed indices must be distinct and within [0, num_ions)")

        self.spin_basis = ['|gg>', '|ge>', '|eg>', '|ee>']
        self.num_spin_states = 4

        self.omega_k = None
        self.b_jk = None
        self.phonon_modes = []
        self.phonon_dims = []
        self.phonon_dim = 1
        self.dim_total = None

        self.detuning = 0.0
        self.max_total_dim = int(max_total_dim)
        self.psi0 = None

        self.omega_k, self.b_jk = self._collective_modes()
        self.update_phonon_modes(importance_threshold=0.01)
        
        # ============================================
        # ADDITION: Initialize correlation tracking
        # ============================================
        self.correlation_results = {
            'entanglement_of_formation': [],
            'quantum_discord': [],
            'mutual_information': [],
            'concurrence': [],
            'purity': [],
            'von_neumann_entropy': []
        }
    
    # [Keep ALL original methods unchanged]
    def _default_ion_positions(self):
        """创建默认的离子位置（线性链）"""
        dz = [5.92296585, 5.43063758, 4.96054013, 4.66658761, 4.45109364, 
              4.27498614, 4.15692247, 4.0974121, 4.06978625, 4.07394696, 
              4.06978625, 4.0974121, 4.15692247, 4.27498614, 4.45109364, 
              4.66658761, 4.96054013, 5.43063758, 5.92296585]
        
        positions = np.zeros(self.num_ions)
        
        for i in range(1, self.num_ions):
            if i-1 < len(dz):
                positions[i] = positions[i-1] + dz[i-1]
            else:
                positions[i] = positions[i-1] + 5.0
        print(positions)
        return positions
    
    def _collective_modes(self):
        """计算集体模式频率和模式向量"""
        L = self.num_ions
        z = self.ion_positions
        omx = self.trap_frequency 
        
        rinv3 = np.abs(np.reshape(z, (1, -1)) - np.reshape(z, (-1, 1)))
        rinv3[range(L), range(L)] = 1
        rinv3 = 1 / rinv3**3
        rinv3[range(L), range(L)] = 0
        
        coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / (1e6)**2
        V = coef * rinv3
        V[range(L), range(L)] = -np.sum(V, axis=1) + omx**2
        
        E, b_jk = np.linalg.eigh(V)
        omega_k = np.sqrt(E)
        print("基于距离和陷阱频率计算的集体模式频率 (MHz):", omega_k/(2*np.pi))
        print(b_jk)
        return omega_k, b_jk
    
    def calculate_importance(self, k):
        """计算模式k的重要性得分"""
        w_k = self.omega_k[k]
        detune = abs(self.detuning)
        g = self.g0
        idx1, idx2 = self.addressed
        b_k1 = self.b_jk[idx1, k]
        b_k2 = self.b_jk[idx2, k]
        importance1 = (g * b_k1)**2 / ((w_k - detune)**2 + (g * b_k1)**2/4)
        importance2 = (g * b_k2)**2 / ((w_k - detune)**2 + (g * b_k2)**2/4)
        importance = np.sqrt(importance1 * importance2)
        return float(np.abs(importance))

    def update_phonon_modes(self, importance_threshold=1e-3, max_modes=5, max_phonons_per_mode=3):
        """Select phonon modes by importance"""
        imps = [self.calculate_importance(k) for k in range(self.num_ions)]
        self.importance_scores = {k: imps[k] for k in range(self.num_ions)}
        sorted_idx = sorted(range(self.num_ions), key=lambda k: imps[k], reverse=True)

        selected = []
        estimated_dim = 1
        for k in sorted_idx:
            if imps[k] < importance_threshold:
                continue
            max_ph = int(max_phonons_per_mode)
            new_dim = estimated_dim * (max_ph + 1)
            if new_dim * self.num_spin_states > self.max_total_dim:
                continue
            coupling1 = self.g0 * self.b_jk[self.addressed[0], k]
            coupling2 = self.g0 * self.b_jk[self.addressed[1], k]
            mode = {
                'frequency': float(self.omega_k[k]),
                'max_phonons': max_ph,
                'coupling1': float(coupling1),
                'coupling2': float(coupling2),
                'initial_occupation': 0,
                'mode_index': int(k),
                'importance': float(imps[k])
            }
            selected.append(mode)
            estimated_dim = new_dim
            if len(selected) >= max_modes:
                break

        if not selected:
            k = sorted_idx[0]
            mode = {
                'frequency': float(self.omega_k[k]),
                'max_phonons': max_phonons_per_mode,
                'coupling1': float(self.g0 * self.b_jk[self.addressed[0], k]),
                'coupling2': float(self.g0 * self.b_jk[self.addressed[1], k]),
                'initial_occupation': 0,
                'mode_index': int(k),
                'importance': float(imps[k])
            }
            selected.append(mode)

        self.phonon_modes = sorted(selected, key=lambda m: m['frequency'])
        self.num_modes = len(self.phonon_modes)
        self.phonon_dims = [m['max_phonons'] + 1 for m in self.phonon_modes]
        self.phonon_dim = int(np.prod(self.phonon_dims))
        self.dim_total = int(self.num_spin_states * self.phonon_dim)

        if self.dim_total > self.max_total_dim:
            raise MemoryError(f"Total dim {self.dim_total} > allowed {self.max_total_dim}")

        self.H = self.build_hamiltonian()
        self.psi0 = None
        print(f"Selected {self.num_modes} modes, total dim = {self.dim_total}")
    
    def set_detuning(self, detuning_MHz):
        """set detuning in MHz (line frequency units); internally convert to rad/μs"""
        self.detuning = float(detuning_MHz) * 2 * np.pi
        print(f"Set detuning = {detuning_MHz} MHz  (internal rad/μs = {self.detuning:.6e})")
        self.update_phonon_modes()

    def set_addressed(self, addressed):
        """change which two ions are addressed (recompute couplings & modes)"""
        if len(addressed) != 2:
            raise ValueError("addressed must be tuple/list of two indices")
        a, b = addressed
        if not (0 <= a < self.num_ions and 0 <= b < self.num_ions and a != b):
            raise ValueError("invalid addressed indices")
        self.addressed = (int(a), int(b))
        print(f"Addressed ions set to {self.addressed}")
        self.update_phonon_modes()

    # [Include all original Hamiltonian building methods unchanged]
    def build_spin_hamiltonian(self):
        sigma_z = diags([1.0, -1.0], 0, shape=(2,2), format='csr')
        s1 = kron(sigma_z, identity(2, format='csr'))
        s2 = kron(identity(2, format='csr'), sigma_z)
        H_spin4 = (self.detuning / 2.0) * (s1 + s2)
        I_ph = identity(self.phonon_dim, format='csr')
        return kron(H_spin4, I_ph)

    def build_phonon_hamiltonian(self):
        H_ph = csr_matrix((self.dim_total, self.dim_total), dtype=complex)
        I_spin4 = identity(self.num_spin_states, format='csr')
        for mode_idx, mode in enumerate(self.phonon_modes):
            n_op = diags(np.arange(0, mode['max_phonons']+1), 0,
                         shape=(mode['max_phonons']+1, mode['max_phonons']+1), format='csr')
            op_list = [identity(d, format='csr') for d in self.phonon_dims]
            op_list[mode_idx] = n_op
            phonon_n = op_list[0]
            for op in op_list[1:]:
                phonon_n = kron(phonon_n, op)
            H_ph += mode['frequency'] * kron(I_spin4, phonon_n)
        return H_ph

    def build_coupling_hamiltonian(self):
        H_cpl = csr_matrix((self.dim_total, self.dim_total), dtype=complex)
        sigma_plus = csr_matrix(np.array([[0,1],[0,0]], dtype=complex))
        sigma_minus = csr_matrix(np.array([[0,0],[1,0]], dtype=complex))

        phonon_ops_a = []
        phonon_ops_adag = []
        for mode_idx, mode in enumerate(self.phonon_modes):
            dim = mode['max_phonons'] + 1
            a_op = diags(np.sqrt(np.arange(1, dim)), -1, shape=(dim, dim), format='csr')
            adag_op = diags(np.sqrt(np.arange(1, dim)), +1, shape=(dim, dim), format='csr')
            op_list_a = [identity(d, format='csr') for d in self.phonon_dims]
            op_list_a[mode_idx] = a_op
            phonon_op_a = op_list_a[0]
            for op in op_list_a[1:]:
                phonon_op_a = kron(phonon_op_a, op)
            op_list_ad = [identity(d, format='csr') for d in self.phonon_dims]
            op_list_ad[mode_idx] = adag_op
            phonon_op_adag = op_list_ad[0]
            for op in op_list_ad[1:]:
                phonon_op_adag = kron(phonon_op_adag, op)
            phonon_ops_a.append(phonon_op_a)
            phonon_ops_adag.append(phonon_op_adag)

        sigma_plus_1 = kron(sigma_plus, identity(2, format='csr'))
        sigma_minus_1 = kron(sigma_minus, identity(2, format='csr'))
        sigma_plus_2 = kron(identity(2, format='csr'), sigma_plus)
        sigma_minus_2 = kron(identity(2, format='csr'), sigma_minus)

        for m_idx, mode in enumerate(self.phonon_modes):
            a_op_full = phonon_ops_a[m_idx]
            adag_op_full = phonon_ops_adag[m_idx]
            g1 = mode['coupling1']
            g2 = mode['coupling2']
            H_cpl += g1 * kron(sigma_plus_1, a_op_full)
            H_cpl += g1 * kron(sigma_minus_1, adag_op_full)
            H_cpl += g2 * kron(sigma_plus_2, a_op_full)
            H_cpl += g2 * kron(sigma_minus_2, adag_op_full)
        return H_cpl

    def build_hamiltonian(self):
        H = self.build_spin_hamiltonian() + self.build_phonon_hamiltonian() + self.build_coupling_hamiltonian()
        return H

    # [Keep all original state preparation and evolution methods]
    def set_mode_occupation(self, mode_index, occ):
        for m in self.phonon_modes:
            if m['mode_index'] == mode_index:
                m['initial_occupation'] = int(occ)
                return
        raise ValueError("invalid mode_index")
    
    def set_initial_state(self, spin_state=None, phonon_occupations=None):
        if isinstance(spin_state, str) and len(spin_state) == 2:
            m = {'g': np.array([1.0,0.0]), 'e': np.array([0.0,1.0])}
            sv = np.kron(m[spin_state[0]], m[spin_state[1]])
        elif spin_state is None:
            sv = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        else:
            sv = np.array(spin_state, dtype=complex)
            if sv.size != 4:
                raise ValueError("spin_state must have length 4 or be 2-char string")

        if phonon_occupations is None:
            phonon_occupations = [m['initial_occupation'] for m in self.phonon_modes]
        if len(phonon_occupations) != self.num_modes:
            raise ValueError("phonon_occupations length mismatch")

        phonon_idx = 0
        factor = 1
        for i in range(self.num_modes-1, -1, -1):
            n = int(phonon_occupations[i])
            dim = self.phonon_dims[i]
            if n >= dim:
                raise ValueError("phonon occupation exceeds truncation")
            phonon_idx += n * factor
            factor *= dim

        psi0 = np.zeros(self.dim_total, dtype=complex)
        for spin_idx, coeff in enumerate(sv):
            global_idx = spin_idx * self.phonon_dim + phonon_idx
            psi0[global_idx] = coeff
        norm = np.linalg.norm(psi0)
        if norm == 0:
            raise ValueError("initial state is zero")
        psi0 /= norm
        self.psi0 = psi0
        self.initial_phonon_idx = phonon_idx
        return psi0

    def evolve(self, t_max, num_points=400, method='RK45', rtol=1e-8, atol=1e-8):
        if self.psi0 is None:
            raise RuntimeError("initial state not set")
        t_span = np.linspace(0.0, float(t_max), int(num_points))
        def rhs(t, psi):
            return -1j * (self.H.dot(psi))
        t0 = time.time()
        sol = solve_ivp(rhs, [0.0, float(t_max)], self.psi0, t_eval=t_span, method=method, rtol=rtol, atol=atol)
        dt = time.time() - t0
        print(f"Evolve done (t_max={t_max}), elapsed {dt:.2f} s")
        return t_span, sol.y

    def compute_expectations(self, psi_t):
        num_pts = psi_t.shape[1]
        res = {'sigma_z1': np.zeros(num_pts), 'sigma_z2': np.zeros(num_pts), 'sigma_zz': np.zeros(num_pts),
               'phonon_numbers': np.zeros((self.num_modes, num_pts))}
        for ti in range(num_pts):
            psi = psi_t[:, ti]
            sz1 = 0.0; sz2 = 0.0; szz = 0.0
            for idx in range(self.dim_total):
                p = np.abs(psi[idx])**2
                spin_idx = idx // self.phonon_dim
                s1 = 0 if (spin_idx // 2) % 2 == 0 else 1
                s2 = 0 if (spin_idx % 2) == 0 else 1
                val1 = 1.0 if s1 == 0 else -1.0
                val2 = 1.0 if s2 == 0 else -1.0
                sz1 += val1 * p
                sz2 += val2 * p
                szz += val1 * val2 * p
            res['sigma_z1'][ti] = sz1
            res['sigma_z2'][ti] = sz2
            res['sigma_zz'][ti] = szz
            for m_idx in range(self.num_modes):
                total_n = 0.0
                for idx in range(self.dim_total):
                    p = np.abs(psi[idx])**2
                    phonon_global_idx = idx % self.phonon_dim
                    tmp = phonon_global_idx
                    for j in range(self.num_modes-1, -1, -1):
                        n_j = tmp % self.phonon_dims[j]
                        tmp //= self.phonon_dims[j]
                        if j == m_idx:
                            total_n += n_j * p
                            break
                res['phonon_numbers'][m_idx, ti] = total_n
        return res
    
    def plot_results(self, t_points, results):
        """绘制结果"""
        plt.figure(figsize=(14, 10))
        gs = GridSpec(3 + self.num_modes, 1)
        
        ax1 = plt.subplot(gs[0])
        plt.plot(t_points, results['sigma_z1'], label=r'$\langle \sigma_{z1} \rangle$')
        plt.plot(t_points, results['sigma_z2'], label=r'$\langle \sigma_{z2} \rangle$')
        plt.plot(t_points, results['sigma_zz'], label=r'$\langle \sigma_{z1} \sigma_{z2} \rangle$')
        plt.ylabel('Spin Expectations')
        plt.grid(True)
        plt.legend()
        plt.title('Two-Ion Spin-Boson Dynamics')
        
        ax2 = plt.subplot(gs[1], sharex=ax1)
        total_phonons = np.sum(results['phonon_numbers'], axis=0)
        plt.plot(t_points, total_phonons, label='Total Phonons', color='purple')
        plt.ylabel('Total Phonons')
        plt.grid(True)
        
        for i in range(self.num_modes):
            ax = plt.subplot(gs[2+i], sharex=ax1)
            mode = self.phonon_modes[i]
            plt.plot(t_points, results['phonon_numbers'][i], 
                     label=f'Mode {mode["mode_index"]}: {mode["frequency"]/(2*np.pi):.2f} MHz')
            plt.ylabel(f'Phonons in Mode {mode["mode_index"]}')
            plt.grid(True)
            plt.legend(fontsize=8)
        
        plt.xlabel('Time (μs)')
        plt.tight_layout()
        plt.show()
        
        print("\nInitial state configuration:")
        print(f"Spin basis: {self.spin_basis}")
        print(f"Initial phonon occupations: {[self.phonon_modes[i]['initial_occupation'] for i in range(self.num_modes)]}")

    def plot_importance(self):
        """绘制所有模式的重要性得分"""
        modes = list(range(self.num_ions))
        importance = [self.importance_scores[k] for k in modes]
        frequencies = [self.omega_k[k] / (2 * np.pi) for k in modes]
        selected_indices = [mode['mode_index'] for mode in self.phonon_modes]
        
        plt.figure(figsize=(12, 6))
        plt.scatter(frequencies, importance, alpha=0.6, label='All modes')
        selected_freq = [freq for i, freq in enumerate(frequencies) if i in selected_indices]
        selected_imp = [imp for i, imp in enumerate(importance) if i in selected_indices]
        plt.scatter(selected_freq, selected_imp, color='red', s=100, label='Selected modes')
        
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Importance score')
        plt.title('Mode Importance Analysis')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ============================================
    # ADDITIONS: New correlation calculation methods
    # ============================================
    
    def get_reduced_density_matrix(self, psi, trace_out='phonons'):
        """Calculate reduced density matrix for the two-spin system"""
        psi_reshaped = psi.reshape(self.num_spin_states, self.phonon_dim)
        
        if trace_out == 'phonons':
            rho_spin = np.zeros((self.num_spin_states, self.num_spin_states), dtype=complex)
            for i in range(self.num_spin_states):
                for j in range(self.num_spin_states):
                    rho_spin[i,j] = np.sum(psi_reshaped[i,:] * np.conj(psi_reshaped[j,:]))
            return rho_spin
        else:
            raise NotImplementedError("Only phonon tracing implemented")
    
    def get_single_spin_density(self, rho_2spin, which_spin):
        """Get single spin density matrix from two-spin density matrix"""
        rho_single = np.zeros((2, 2), dtype=complex)
        
        if which_spin == 0:
            rho_single[0,0] = rho_2spin[0,0] + rho_2spin[1,1]
            rho_single[0,1] = rho_2spin[0,2] + rho_2spin[1,3]
            rho_single[1,0] = rho_2spin[2,0] + rho_2spin[3,1]
            rho_single[1,1] = rho_2spin[2,2] + rho_2spin[3,3]
        else:
            rho_single[0,0] = rho_2spin[0,0] + rho_2spin[2,2]
            rho_single[0,1] = rho_2spin[0,1] + rho_2spin[2,3]
            rho_single[1,0] = rho_2spin[1,0] + rho_2spin[3,2]
            rho_single[1,1] = rho_2spin[1,1] + rho_2spin[3,3]
            
        return rho_single
    
    def calculate_concurrence(self, rho):
        """Calculate concurrence for a two-qubit density matrix"""
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_y_2qubit = np.kron(sigma_y, sigma_y)
        
        rho_tilde = sigma_y_2qubit @ np.conj(rho) @ sigma_y_2qubit
        sqrt_rho = sqrtm(rho)
        R = sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)
        
        eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
        C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
        return float(np.real(C))
    
    def calculate_entanglement_of_formation(self, rho):
        """Calculate entanglement of formation"""
        C = self.calculate_concurrence(rho)
        
        if C == 0:
            return 0.0
        
        def H(x):
            if x == 0 or x == 1:
                return 0
            return -x * np.log2(x) - (1-x) * np.log2(1-x)
        
        x = (1 + np.sqrt(1 - C**2)) / 2
        return H(x)
    
    def von_neumann_entropy(self, rho):
        """Calculate von Neumann entropy"""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def calculate_mutual_information(self, rho_AB, rho_A, rho_B):
        """Calculate quantum mutual information"""
        S_A = self.von_neumann_entropy(rho_A)
        S_B = self.von_neumann_entropy(rho_B)
        S_AB = self.von_neumann_entropy(rho_AB)
        return S_A + S_B - S_AB
    
    def calculate_quantum_discord(self, rho, num_measurements=100):
        """Calculate quantum discord using optimization over measurements"""
        rho_A = self.get_single_spin_density(rho, 0)
        rho_B = self.get_single_spin_density(rho, 1)
        
        I_total = self.calculate_mutual_information(rho, rho_A, rho_B)
        
        max_classical_corr = 0
        
        for _ in range(num_measurements):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            
            n = np.array([np.sin(theta)*np.cos(phi), 
                         np.sin(theta)*np.sin(phi), 
                         np.cos(theta)])
            
            P0 = (np.eye(2) + n[0]*np.array([[0,1],[1,0]]) + 
                  n[1]*np.array([[0,-1j],[1j,0]]) + 
                  n[2]*np.array([[1,0],[0,-1]])) / 2
            P1 = np.eye(2) - P0
            
            Pi_0 = np.kron(np.eye(2), P0)
            Pi_1 = np.kron(np.eye(2), P1)
            
            rho_0 = Pi_0 @ rho @ Pi_0.conj().T
            p_0 = np.real(np.trace(rho_0))
            
            rho_1 = Pi_1 @ rho @ Pi_1.conj().T
            p_1 = np.real(np.trace(rho_1))
            
            classical_corr = 0
            if p_0 > 1e-10:
                rho_0_norm = rho_0 / p_0
                rho_A_0 = self.get_single_spin_density(rho_0_norm, 0)
                S_A_0 = self.von_neumann_entropy(rho_A_0)
                classical_corr += p_0 * (self.von_neumann_entropy(rho_A) - S_A_0)
            
            if p_1 > 1e-10:
                rho_1_norm = rho_1 / p_1
                rho_A_1 = self.get_single_spin_density(rho_1_norm, 0)
                S_A_1 = self.von_neumann_entropy(rho_A_1)
                classical_corr += p_1 * (self.von_neumann_entropy(rho_A) - S_A_1)
            
            max_classical_corr = max(max_classical_corr, classical_corr)
        
        discord = I_total - max_classical_corr
        return max(0, discord)
    
    def calculate_purity(self, rho):
        """Calculate purity Tr(ρ²)"""
        return np.real(np.trace(rho @ rho))
    
    def compute_expectations_with_correlations(self, psi_t):
        """Extended version computing both expectations and correlations"""
        num_pts = psi_t.shape[1]
        results = self.compute_expectations(psi_t)
        
        results['entanglement'] = np.zeros(num_pts)
        results['discord'] = np.zeros(num_pts)
        results['mutual_info'] = np.zeros(num_pts)
        results['concurrence'] = np.zeros(num_pts)
        results['purity'] = np.zeros(num_pts)
        results['entropy'] = np.zeros(num_pts)
        
        for ti in range(num_pts):
            psi = psi_t[:, ti]
            rho_spin = self.get_reduced_density_matrix(psi)
            
            results['entanglement'][ti] = self.calculate_entanglement_of_formation(rho_spin)
            results['concurrence'][ti] = self.calculate_concurrence(rho_spin)
            results['discord'][ti] = self.calculate_quantum_discord(rho_spin, num_measurements=50)
            results['purity'][ti] = self.calculate_purity(rho_spin)
            results['entropy'][ti] = self.von_neumann_entropy(rho_spin)
            
            rho_A = self.get_single_spin_density(rho_spin, 0)
            rho_B = self.get_single_spin_density(rho_spin, 1)
            results['mutual_info'][ti] = self.calculate_mutual_information(rho_spin, rho_A, rho_B)
        
        return results
    
    def plot_results_with_correlations(self, t_points, results):
        """Enhanced plotting including correlation dynamics"""

        FIGURE_DIR = "C:/Users/86182/Desktop/yidong/2025_project/Paper_proposal/preservation of quantum correlation between spin and phonon/Figure"
        os.makedirs(FIGURE_DIR, exist_ok=True)

        plt.figure(figsize=(16, 12))
        gs = GridSpec(5 + self.num_modes, 2, width_ratios=[1, 1])
        
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(t_points, results['sigma_z1'], label=r'$\langle \sigma_{z1} \rangle$')
        ax1.plot(t_points, results['sigma_z2'], label=r'$\langle \sigma_{z2} \rangle$')
        ax1.plot(t_points, results['sigma_zz'], label=r'$\langle \sigma_{z1} \sigma_{z2} \rangle$')
        ax1.set_ylabel('Spin Expectations')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('Spin Dynamics')
        
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(t_points, results['entanglement'], label='EoF', color='red', linewidth=2)
        ax2.plot(t_points, results['concurrence'], label='Concurrence', color='orange', linestyle='--')
        ax2.set_ylabel('Entanglement Measures')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title('Entanglement Dynamics')
        
        ax3 = plt.subplot(gs[1, 0], sharex=ax1)
        total_phonons = np.sum(results['phonon_numbers'], axis=0)
        ax3.plot(t_points, total_phonons, color='purple', linewidth=2)
        ax3.set_ylabel('Total Phonons')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Phonon Excitation')
        
        ax4 = plt.subplot(gs[1, 1], sharex=ax2)
        ax4.plot(t_points, results['discord'], label='Quantum Discord', color='blue', linewidth=2)
        ax4.plot(t_points, results['mutual_info'], label='Mutual Info', color='cyan', linestyle='--')
        ax4.set_ylabel('Information Measures')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_title('Quantum Correlations')
        
        ax5 = plt.subplot(gs[2, 0], sharex=ax1)
        ax5.plot(t_points, results['purity'], label='Purity', color='green')
        ax5.plot(t_points, results['entropy'], label='Von Neumann Entropy', color='darkgreen', linestyle='--')
        ax5.set_ylabel('State Measures')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.set_title('Mixed State Characterization')
        
        ax6 = plt.subplot(gs[2, 1], sharex=ax2)
        ax6.plot(t_points, results['entanglement'], label='EoF', alpha=0.7)
        ax6.plot(t_points, results['discord'], label='Discord', alpha=0.7)
        ax6.fill_between(t_points, 0, results['discord'] - results['entanglement'], 
                         where=(results['discord'] > results['entanglement']), 
                         alpha=0.3, label='Discord > EoF')
        ax6.set_ylabel('Correlation Measures')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        ax6.set_title('EoF vs Discord')
        
        for i in range(self.num_modes):
            ax = plt.subplot(gs[3+i, :])
            mode = self.phonon_modes[i]
            ax.plot(t_points, results['phonon_numbers'][i])
            ax.set_ylabel(f'Mode {mode["mode_index"]}')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title('Individual Phonon Mode Dynamics')
        
        ax.set_xlabel('Time (μs)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'dynamics_evolution.pdf'))
        plt.show()
        
        print("\n" + "="*60)
        print("QUANTUM CORRELATION SUMMARY")
        print("="*60)
        print(f"Maximum Entanglement of Formation: {np.max(results['entanglement']):.4f} ebits")
        print(f"Average Entanglement of Formation: {np.mean(results['entanglement']):.4f} ebits")
        print(f"Maximum Concurrence: {np.max(results['concurrence']):.4f}")
        print(f"Maximum Quantum Discord: {np.max(results['discord']):.4f}")
        print(f"Average Quantum Discord: {np.mean(results['discord']):.4f}")
        print(f"Discord exceeds EoF for {np.sum(results['discord'] > results['entanglement'])/len(t_points)*100:.1f}% of evolution")
        print(f"Minimum Purity: {np.min(results['purity']):.4f}")
        print(f"Maximum Von Neumann Entropy: {np.max(results['entropy']):.4f} bits")


# Main execution
if __name__ == "__main__":
    trap_config = {'num_ions': 20, 'trap_frequency': 2.4057, 'ion_positions': None}
    system = TwoSpinsInChain(trap_config, addressed=(0,1), g0=0.0067, max_total_dim=500000)
    
    print("Selected phonon modes:")
    for mode in system.phonon_modes:
        print(f"Mode {mode['mode_index']}: Frequency={mode['frequency']/(2*np.pi):.2f}MHz, "
              f"Coupling1={mode['coupling1']/(2*np.pi):.4f}MHz, Coupling2={mode['coupling2']/(2*np.pi):.4f}MHz, "
              f"Importance={mode['importance']:.4f}, Max phonons={mode['max_phonons']}")
    
    system.set_detuning(-2.17680521)
    system.plot_importance()
    
    system.set_initial_state(spin_state=[0.0, 0.0, 0.0, 1.0]) 
    
    t_points, psi_t = system.evolve(t_max=1000, num_points=1000)
    
    # Use the enhanced correlation computation
    results = system.compute_expectations_with_correlations(psi_t)
    
    # Use the enhanced plotting
    system.plot_results_with_correlations(t_points, results)
    
    np.savez('two_ion_results_with_correlations.npz', 
             t=t_points, 
             psi=psi_t,
             sigma_z1=results['sigma_z1'],
             sigma_z2=results['sigma_z2'],
             sigma_zz=results['sigma_zz'],
             phonon_numbers=results['phonon_numbers'],
             entanglement=results['entanglement'],
             concurrence=results['concurrence'],
             discord=results['discord'],
             mutual_info=results['mutual_info'],
             purity=results['purity'],
             entropy=results['entropy'],
             trap_config=trap_config,
             phonon_modes=system.phonon_modes)