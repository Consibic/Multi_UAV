import numpy as np

def get_z_comp(l, r):
    r_sq = r[0]**2 + r[1]**2
    return np.sqrt(np.abs(l**2 - r_sq))

def get_B(l, r):
    z_comp = get_z_comp(l, r)
    B = np.array([
        [1, 0],
        [0, 1],
        [r[0] / z_comp, r[1] / z_comp]
    ])
    return B

def get_B_dot(l, r, v_p):
    B_dot_upper = np.zeros((2, 2))
    r_transpose = r.reshape(1, 2)
    v_p_transpose = v_p.reshape(1, 2)
    z_comp = get_z_comp(l, r)
    B_dot_lower = (v_p_transpose * z_comp**2 + 
                   (r_transpose @ v_p) @ r_transpose) / z_comp**3
    B_dot = np.vstack([B_dot_upper, B_dot_lower])
    return B_dot

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def get_A(tether, mass):
    A = np.zeros((3, 3))
    for j in range(len(mass)):
        t_j_skew = skew_symmetric(tether[:, j])
        A += t_j_skew * mass[j]
    return A

def get_J(tether, mass):
    J = np.zeros((3, 3))
    for j in range(len(mass)):
        t_j_skew = skew_symmetric(tether[:, j])
        J += -mass[j] * t_j_skew @ t_j_skew
    return J