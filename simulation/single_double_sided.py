# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# Hardware Parameters
# ==========================================
dz_array = np.array([11.1628, 9.086, 8.6671, 8.6553, 8.6317, 9.0211, 10.6613])
f_com_kHz = 1506.8
f_k_exp = np.array([1461.4, 1470.8, 1479.4, 1486.8, 1493.8, 1499.4, 1503.8, 1506.8])

def collective_mode(omx_rad, dz):
    num = len(dz) + 1
    z = np.zeros(num)
    for i in range(num - 1): z[i+1] = z[i] + dz[i]
    L = len(z)
    rinv3 = np.abs(np.reshape(z, (1, -1)) - np.reshape(z, (-1, 1)))
    np.fill_diagonal(rinv3, 1)
    rinv3 = 1 / rinv3**3
    np.fill_diagonal(rinv3, 0)
    coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / 1e3**2
    V = coef * rinv3
    V[range(L), range(L)] = -np.sum(V, axis=1) + omx_rad**2
    E, b_jk = np.linalg.eigh(V)
    return np.sqrt(np.maximum(E, 1e-10)), b_jk

omx_rad = 2 * np.pi * f_com_kHz
_, b_matrix = collective_mode(omx_rad, dz_array)
N_ions = 8

ion_A = 3  # Index 3 is Ion 4
ion_B = 5  # Index 5 is Ion 6

b_A_eff = b_matrix[ion_A, :] * np.sqrt(f_com_kHz / f_k_exp)
b_B_eff = b_matrix[ion_B, :] * np.sqrt(f_com_kHz / f_k_exp)

g_strength = 2 * np.pi * 8.0 # 8.0 kHz

# ==========================================
# Dynamics Functions
# ==========================================
def run_single_sided(target_freq_kHz, T_max_us=300):
    omega_k = 2 * np.pi * (f_k_exp - target_freq_kHz)

    def odefun(t, y):
        c_uu = y[0]
        c_k = y[1:]
        dy = np.zeros(1 + N_ions, dtype=np.complex128)

        for k in range(N_ions):
            H_Ak = g_strength * b_A_eff[k]
            dy[0] += -1j * H_Ak * c_k[k]
            dy[k+1] += -1j * np.conj(H_Ak) * c_uu - 1j * omega_k[k] * c_k[k]
        return dy

    y0 = np.zeros(1 + N_ions, dtype=np.complex128)
    y0[0] = 1.0 # The c_uu component inside the 1/sqrt(2) brace

    t_eval = np.linspace(0, T_max_us/1000.0, 5000)
    sol = solve_ivp(odefun, [0, T_max_us/1000.0], y0, t_eval=t_eval, max_step=0.0005, atol=1e-8, rtol=1e-8)

    c_uu_t = sol.y[0]
    P_t = np.abs(c_uu_t)**2
    C_t = np.abs(c_uu_t) # Exact C = sqrt(P) relation

    return sol.t * 1000.0, P_t, C_t

def run_double_sided(target_freq_kHz, T_max_us=300):
    """
    Simulates the |uu> decay in a double-sided coupling model using QuTiP.
    """
    import qutip as qt

    # 2 spins, N modes, truncated at 2 phonons each
    N_ph = 2

    I_spin = qt.qeye(2)
    sig_p = qt.sigmap()
    sig_m = qt.sigmam()

    sm_A = qt.tensor(sig_m, I_spin)
    sp_A = qt.tensor(sig_p, I_spin)
    sz_A = qt.tensor(qt.sigmaz(), I_spin)

    sm_B = qt.tensor(I_spin, sig_m)
    sp_B = qt.tensor(I_spin, sig_p)
    sz_B = qt.tensor(I_spin, qt.sigmaz())

    # Modes
    a_ops = []
    for k in range(N_ions):
        a = qt.destroy(N_ph)
        # Tensor it properly: 2 spins + N modes
        op_list = [I_spin, I_spin] + [qt.qeye(N_ph)] * N_ions
        op_list[2 + k] = a
        a_ops.append(qt.tensor(*op_list))

    # Upgrade spin ops
    sp_A_full = qt.tensor([sp_A] + [qt.qeye(N_ph)] * N_ions)
    sm_A_full = qt.tensor([sm_A] + [qt.qeye(N_ph)] * N_ions)
    sp_B_full = qt.tensor([sp_B] + [qt.qeye(N_ph)] * N_ions)
    sm_B_full = qt.tensor([sm_B] + [qt.qeye(N_ph)] * N_ions)

    sz_A_full = qt.tensor([sz_A] + [qt.qeye(N_ph)] * N_ions)
    sz_B_full = qt.tensor([sz_B] + [qt.qeye(N_ph)] * N_ions)

    # Hamiltonian
    omega_k = 2 * np.pi * (f_k_exp - target_freq_kHz)

    H = 0
    for k in range(N_ions):
        H += omega_k[k] * a_ops[k].dag() * a_ops[k]

        H_Ak = g_strength * b_A_eff[k]
        H_Bk = g_strength * b_B_eff[k]

        # Interaction (Rotating wave approximation, sigma_+ a + sigma_- a^dag)
        H += H_Ak * (sp_A_full * a_ops[k] + sm_A_full * a_ops[k].dag())
        H += H_Bk * (sp_B_full * a_ops[k] + sm_B_full * a_ops[k].dag())

    # Initial state |Phi+> = 1/sqrt(2) (|uu> + |dd>) |vac>
    up = qt.basis(2, 0)
    down = qt.basis(2, 1)

    uu = qt.tensor(up, up)
    dd = qt.tensor(down, down)
    phi_plus = (uu + dd) / np.sqrt(2)

    psi0 = qt.tensor([phi_plus] + [qt.basis(N_ph, 0)] * N_ions)

    tlist = np.linspace(0, T_max_us/1000.0, 500) # Faster eval for testing

    # Solve dynamics
    result = qt.sesolve(H, psi0, tlist, [])

    P_t = []
    C_t = []

    # Parity operator = sigma_z^A * sigma_z^B
    sz_A_op = qt.tensor(qt.sigmaz(), I_spin)
    sz_B_op = qt.tensor(I_spin, qt.sigmaz())
    Parity_op = sz_A_op * sz_B_op

    for state in result.states:
        rho_spin = state.ptrace([0, 1])

        # Calculate Parity and Concurrence based on User prompt definitions
        # The user's definition of Parity for Double-Sided:
        # User prompt: "P can be negative, meaning ESD happened... <P>_ss drops to ~0.13"
        # Often Parity P = 2(rho_uu + rho_dd) - 1 or something.
        # But they explicitly defined for single sided: P(t) = |c_uu(t)|^2
        # And Bell fidelity: F(t) = (|c_uu|^2/2 + 1/2 + Re[c_uu]) / 2
        # For double sided, we will just use the exact Concurrence and trace-based Parity

        # Let's extract the actual Bell elements from the reduced density matrix
        # rho_spin basis: |uu>, |ud>, |du>, |dd> (QuTiP standard 00, 01, 10, 11)
        # Note: |uu> = 0, |ud> = 1, |du> = 2, |dd> = 3
        # Concurrence formula: max(0, 2|rho_ud| - 2 sqrt(rho_uu rho_dd)) ? No, for X state:
        # C = 2 max(0, |rho_03| - sqrt(rho_11 rho_22), |rho_12| - sqrt(rho_00 rho_33))
        # Here we use QuTiP's builtin concurrence
        concurrence = qt.concurrence(rho_spin)
        C_t.append(concurrence)

        # User's Parity P is likely the coherence + populations.
        # In many setups, P = <sigma_z_A sigma_z_B> or similar.
        # But user says: P = |c_uu|^2 for single sided.
        # Actually in Bell state experiments, Parity P = P_uu + P_dd - P_ud - P_du.
        # Let's use this definition!
        P_uu = np.real(rho_spin[0, 0])
        P_ud = np.real(rho_spin[1, 1])
        P_du = np.real(rho_spin[2, 2])
        P_dd = np.real(rho_spin[3, 3])
        Parity = P_uu + P_dd - P_ud - P_du
        P_t.append(Parity)

    return tlist * 1000.0, np.array(P_t), np.array(C_t)

if __name__ == "__main__":
    print("Testing Single-Sided...")
    t_s, P_s, C_s = run_single_sided(1463.0, 10) # 10us
    print("Single-Sided test done.")

    print("Testing Double-Sided...")
    t_d, P_d, C_d = run_double_sided(1463.0, 10) # 10us
    print("Double-Sided test done.")
