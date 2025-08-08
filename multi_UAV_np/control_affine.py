import numpy as np
from scipy.linalg import block_diag
from Multi_UAV.multi_UAV_np.kinematics import *
from Multi_UAV.multi_UAV_np.system_matrices import *

# Mass of the payload
m_p = 0.5

# Mass of every drone
Mass = [1.5, 1.5, 1.5]

# Length of every cable
Cable = [1.5, 1.5, 1.5]

# Tether point vector of every drone in body frame
Tether = np.array([[-0.5, -0.5, 0.5],
                   [0.5, 0.5, 0.5],
                   [0.0, 0.5, 0.5]])
tether_1 = Tether[:, 0]
tether_2 = Tether[:, 1]
tether_3 = Tether[:, 2]

# Payload moment of inertia
J_p = np.diag([0.0408, 0.1, 0.1])

# Gravity constant
g = 9.81

def get_fx(x):
    
    # Unpack state (24 states)
    r_p_x, r_p_y, r_p_z, alpha, beta, gamma, r_q1_x, r_q1_y, r_q2_x, r_q2_y, r_q3_x, r_q3_y, v_p_x, v_p_y, v_p_z, omega_x, omega_y, omega_z, v_q1_x, v_q1_y, v_q2_x, v_q2_y, v_q3_x, v_q3_y = x
    r_p = np.array([r_p_x, r_p_y, r_p_z])
    r_q1 = np.array([r_q1_x, r_q1_y])
    r_q2 = np.array([r_q2_x, r_q2_y])
    r_q3 = np.array([r_q3_x, r_q3_y])
    omega = np.array([omega_x, omega_y, omega_z])
    v_p = np.array([v_p_x, v_p_y, v_p_z])
    v_p_xy = np.array([v_p_x, v_p_y])
    v_q1 = np.array([v_q1_x, v_q1_y])
    v_q2 = np.array([v_q2_x, v_q2_y])
    v_q3 = np.array([v_q3_x, v_q3_y])
    # Rotation matrix from body to inertial frame
    R11 = np.cos(beta) * np.cos(gamma)
    R12 = -np.cos(beta) * np.sin(gamma)
    R13 = np.sin(beta)
    R21 = np.cos(alpha) * np.sin(gamma) + np.sin(alpha) * np.sin(beta) * np.cos(gamma)
    R22 = np.cos(alpha) * np.cos(gamma) - np.sin(alpha) * np.sin(beta) * np.sin(gamma)
    R23 = -np.sin(alpha) * np.cos(beta)
    R31 = np.sin(alpha) * np.sin(gamma) - np.cos(alpha) * np.sin(beta) * np.cos(gamma)
    R32 = np.sin(alpha) * np.cos(gamma) + np.cos(alpha) * np.sin(beta) * np.sin(gamma)
    R33 = np.cos(alpha) * np.cos(beta)
    R_IP = np.array([[R11, R12, R13],
                  [R21, R22, R23],
                  [R31, R32, R33]])
    R_PI = R_IP.T

    # Matrix B
    B_1 = get_B(Cable[0], r_q1)
    B_2 = get_B(Cable[1], r_q2)
    B_3 = get_B(Cable[2], r_q3)
    B_1_T = B_1.T
    B_2_T = B_2.T
    B_3_T = B_3.T

    # Matrix B_dot
    B_1_dot = get_B_dot(Cable[0], r_q1, v_q1)
    B_2_dot = get_B_dot(Cable[1], r_q2, v_q2)
    B_3_dot = get_B_dot(Cable[2], r_q3, v_q3)

    # Skew-symmetric matrix for the payload
    t_1_skew = skew_symmetric(Tether[:, 0])
    t_2_skew = skew_symmetric(Tether[:, 1])
    t_3_skew = skew_symmetric(Tether[:, 2])
    omega_skew = skew_symmetric(omega)

    # Matrix A
    A = get_A(Tether, Mass)
    A_T = A.T

    # Matrix J_q
    J_q = get_J(Tether, Mass)
    g_I = np.array([0, 0, g]).reshape(3, 1)

    velocities = np.concatenate([v_p, omega, v_q1, v_q2, v_q3]).reshape(-1, 1)

    # Get system matrices
    M = get_M(x)
    C = get_C(x)
    G = get_G(x)

    # Get fx
    fx_upper = velocities
    fx_lower = np.linalg.solve(M, G - C @ velocities)
    fx = np.vstack([fx_upper, fx_lower])

    return fx

def get_gx(x):

    # Unpack state (24 states)
    r_p_x, r_p_y, r_p_z, alpha, beta, gamma, r_q1_x, r_q1_y, r_q2_x, r_q2_y, r_q3_x, r_q3_y, v_p_x, v_p_y, v_p_z, omega_x, omega_y, omega_z, v_q1_x, v_q1_y, v_q2_x, v_q2_y, v_q3_x, v_q3_y = x
    r_p = np.array([r_p_x, r_p_y, r_p_z])
    r_q1 = np.array([r_q1_x, r_q1_y])
    r_q2 = np.array([r_q2_x, r_q2_y])
    r_q3 = np.array([r_q3_x, r_q3_y])
    omega = np.array([omega_x, omega_y, omega_z])
    v_p = np.array([v_p_x, v_p_y, v_p_z])
    v_p_xy = np.array([v_p_x, v_p_y])
    v_q1 = np.array([v_q1_x, v_q1_y])
    v_q2 = np.array([v_q2_x, v_q2_y])
    v_q3 = np.array([v_q3_x, v_q3_y])
    # Rotation matrix from body to inertial frame
    R11 = np.cos(beta) * np.cos(gamma)
    R12 = -np.cos(beta) * np.sin(gamma)
    R13 = np.sin(beta)
    R21 = np.cos(alpha) * np.sin(gamma) + np.sin(alpha) * np.sin(beta) * np.cos(gamma)
    R22 = np.cos(alpha) * np.cos(gamma) - np.sin(alpha) * np.sin(beta) * np.sin(gamma)
    R23 = -np.sin(alpha) * np.cos(beta)
    R31 = np.sin(alpha) * np.sin(gamma) - np.cos(alpha) * np.sin(beta) * np.cos(gamma)
    R32 = np.sin(alpha) * np.cos(gamma) + np.cos(alpha) * np.sin(beta) * np.sin(gamma)
    R33 = np.cos(alpha) * np.cos(beta)
    R_IP = np.array([[R11, R12, R13],
                  [R21, R22, R23],
                  [R31, R32, R33]])
    R_PI = R_IP.T

    # Matrix B
    B_1 = get_B(Cable[0], r_q1)
    B_2 = get_B(Cable[1], r_q2)
    B_3 = get_B(Cable[2], r_q3)
    B_1_T = B_1.T
    B_2_T = B_2.T
    B_3_T = B_3.T

    # Matrix B_dot
    B_1_dot = get_B_dot(Cable[0], r_q1, v_q1)
    B_2_dot = get_B_dot(Cable[1], r_q2, v_q2)
    B_3_dot = get_B_dot(Cable[2], r_q3, v_q3)

    # Skew-symmetric matrix for the payload
    t_1_skew = skew_symmetric(Tether[:, 0])
    t_2_skew = skew_symmetric(Tether[:, 1])
    t_3_skew = skew_symmetric(Tether[:, 2])
    omega_skew = skew_symmetric(omega)

    # Matrix A
    A = get_A(Tether, Mass)
    A_T = A.T

    # Matrix J_q
    J_q = get_J(Tether, Mass)
    g_I = np.array([0, 0, g]).reshape(3, 1)

    # Get system matrices
    M = get_M(x)

    # Control input matrix H (12x9)
    H_lower = block_diag(B_1_T, B_2_T, B_3_T)
    H = np.vstack([np.zeros((6, 9)), H_lower])

    gx_upper = np.zeros((12, 9))
    gx_lower = np.linalg.solve(M, H)

    gx = np.vstack([gx_upper, gx_lower])

    return gx