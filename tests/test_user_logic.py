import numpy as np

def test_user_logic():
    ion = 0
    L = 10

    dz = [6.0019812,5.15572782,4.72673127,4.54681994,4.47707492,4.54681994,4.72673127,5.15572782,6.0019812]

    omega = 2*np.pi*np.array([242412, 242401, 242389, 242374, 242356, 242335, 242311, 242284, 242255, 242223]) # 12.24 10ions

    omega_k = omega - 2.0 * np.pi * 240.0203e3
    omega_k = np.sort(omega_k)

    omx = np.max(omega) - 2.0 * np.pi * 240.0203e3 # kHz
    # Note: max(omega) is roughly 2*pi*242412. Ref is 2*pi*240020.3.
    # Difference is 2*pi*2391.7.
    # If the comment says # kHz, maybe they mean omx corresponds to kHz value?
    # But it is calculated in angular frequency (rad/s) because of 2*pi factors.

    num = int(len(dz)+1)
    z = np.zeros(num)
    for i in range(num-1):
        z[i+1] = z[i] + dz[i]

    def collective_mode(omx, z):
        '''
        Return collective mode frequency omega_k w.r.t. highest mode
        and mode vectors b_jk
        '''
        L = len(z)
        rinv3 = np.abs(np.reshape(z, (1, -1)) - np.reshape(z, (-1, 1)))
        rinv3[range(L), range(L)] = 1
        rinv3 = 1 / rinv3**3
        rinv3[range(L), range(L)] = 0
        coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / 1e3**2
        V = coef * rinv3
        V[range(L), range(L)] = -np.sum(V, axis=1) + omx**2
        E, b_jk = np.linalg.eigh(V)
        omega_k_calc = np.sqrt(E)
        return(omega_k_calc - np.max(omega_k_calc), b_jk)

    print(f"Input omx (rad/s): {omx}")
    # The function expects omx to be compatible with 'coef'.
    # coef ~ 10^9.
    # omx (rad/s) ~ 1.5e4. omx^2 ~ 2e8.
    # 2e8 vs 10^9. Comparable!
    # So the unit consistency relies on omx being ~10^4.
    # If omx was 242 kHz (linear) -> 2.4e5. omx^2 ~ 6e10.
    # Then omx^2 >> coef.

    omega_k_cal, b_jk = collective_mode(omx, z)
    print("Calculated omega_k (relative):", omega_k_cal)
    print("Provided omega_k (relative):", omega_k - np.max(omega_k))

    # Check absolute values
    # In collective_mode: V = coef*rinv3 + omx^2
    # E = eigenvalues(V).
    # If coef ~ 10^9 and omx^2 ~ 2.25e8.
    # Then E ~ 10^9.
    # sqrt(E) ~ 30000.
    # 30000 is close to 2*pi*2400 (~15000) or 2*pi*240000 (~1.5e6)?
    # 30000 is 3e4.
    # 2*pi*2400 ~ 1.5e4.
    # It's in the ballpark of the *shifted* frequencies (kHz range).

    print(f"Max calculated freq (sqrt(E)): {np.max(omega_k_cal + np.max(np.sqrt(np.linalg.eigh(coef * 1/np.abs(np.subtract.outer(z,z)+np.eye(L))**3 - np.diag(np.sum(coef * 1/np.abs(np.subtract.outer(z,z)+np.eye(L))**3, axis=1)) + omx**2)[0])))}")

    # Re-calculate E explicitly to see absolute magnitude
    rinv3 = np.abs(np.reshape(z, (1, -1)) - np.reshape(z, (-1, 1)))
    rinv3[range(L), range(L)] = 1
    rinv3 = 1 / rinv3**3
    rinv3[range(L), range(L)] = 0
    coef = 9e9 * 1.602e-19**2 / 171 / 1.66e-27 / 1e-6**3 / 1e3**2
    V = coef * rinv3
    V[range(L), range(L)] = -np.sum(V, axis=1) + omx**2
    E, _ = np.linalg.eigh(V)
    freqs = np.sqrt(E)
    print("Absolute Calculated Freqs:", freqs)
    print("Reference 'omx':", omx)

if __name__ == "__main__":
    test_user_logic()
