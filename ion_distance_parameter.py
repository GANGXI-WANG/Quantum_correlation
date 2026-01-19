import numpy as np
import matplotlib.pyplot as plt
import sys,os
import time

ion = 0
L = 10

dz = [6.0019812,5.15572782,4.72673127,4.54681994,4.47707492,4.54681994,4.72673127,5.15572782,6.0019812]

omega = 2*np.pi*np.array([242412, 242401, 242389, 242374, 242356, 242335, 242311, 242284, 242255, 242223]) # 12.24 10ions

omega_k = omega - 2.0 * np.pi * 240.0203e3
omega_k = np.sort(omega_k)

omx = max(omega) - 2.0 * np.pi * 240.0203e3 # kHz

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
    omega_k = np.sqrt(E)
    return(omega_k - np.max(omega_k), b_jk)
omega_k_cal, b_jk = collective_mode(omx, z)
print(omega_k)
print(b_jk)
b_k = b_jk[ion, :]
print(b_k)

# temtime = time.strftime('%Y-%m-%d-%H-%M-%S')
# name = '10ions.12.24data_Load2' + str(temtime)
# BASE_PATH = "E:/Gangxi/20220818py-script/experience_data"
# np.savez(os.path.join(BASE_PATH, name), L=L, omx=omx, omega_k=omega_k, b_jk=b_jk)