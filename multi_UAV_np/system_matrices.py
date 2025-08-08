import numpy as np
from Multi_UAV.multi_UAV_np.kinematics import *

# Note that our control input is 9x1, basically x,y,z lift control for each drone

# Mass of the payload
m_p = 1.3

# Mass of every drone
Mass = [1.63, 1.63, 1.63]

# Length of every cable
Cable = [0.98, 0.98, 0.98]

# Tether point vector of every drone in body frame
Tether = np.array([[1.085, 0, 0],  # cable anchor points
                   [-0.5425, -0.9396, 0],
                   [-0.5425, 0.9396, 0]])
tether_1 = Tether[:, 0]
tether_2 = Tether[:, 1]
tether_3 = Tether[:, 2]

# Payload moment of inertia
J_p = np.diag([0.0408, 0.1, 0.1])

# Gravity constant
g = 9.81

def get_M(x):

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

    # Mass matrix M (12x12)
    M11 = (m_p + np.sum(Mass)) * np.eye(3)
    M12 = -R_PI @ A_T
    M13 = Mass[0] * B_1
    M14 = Mass[1] * B_2
    M15 = Mass[2] * B_3
    M22 = J_p + J_q
    M23 = Mass[0] * (t_1_skew @ (R_PI @ B_1))
    M24 = Mass[1] * (t_2_skew @ (R_PI @ B_2))
    M25 = Mass[2] * (t_3_skew @ (R_PI @ B_3))
    M33 = Mass[0] * (B_1_T @ B_1)
    M44 = Mass[1] * (B_2_T @ B_2)
    M55 = Mass[2] * (B_3_T @ B_3)
    
    # Assemble the full mass matrix
    M = np.block([
        [M11, M12, M13, M14, M15],
        [M12.T, M22, M23, M24, M25],
        [M13.T, M23.T, M33, np.zeros((2, 2)), np.zeros((2, 2))],
        [M14.T, M24.T, np.zeros((2, 2)), M44, np.zeros((2, 2))],
        [M15.T, M25.T, np.zeros((2, 2)), np.zeros((2, 2)), M55]
    ])

    return M

def get_C(x):

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

    # Coriolis matrix C (12x12)
    C_left = np.zeros((12, 3))
    C12 = R_IP @ (omega_skew @ A_T)
    C13 = Mass[0] * B_1_dot
    C14 = Mass[1] * B_2_dot
    C15 = Mass[2] * B_3_dot
    C22 = -skew_symmetric(J_p @ omega) - ((Mass[0] * (t_1_skew @ (omega_skew @ t_1_skew))) +
                                          (Mass[1] * (t_2_skew @ (omega_skew @ t_2_skew))) +
                                          (Mass[2] * (t_3_skew @ (omega_skew @ t_3_skew)))
                                         )
    C23 = Mass[0] * (t_1_skew @ (R_PI @ B_1_dot))
    C24 = Mass[1] * (t_2_skew @ (R_PI @ B_2_dot))
    C25 = Mass[2] * (t_3_skew @ (R_PI @ B_3_dot))
    C32 = -Mass[0] * (B_1_T @ (R_IP @ (omega_skew @ t_1_skew)))
    C42 = -Mass[1] * (B_2_T @ (R_IP @ (omega_skew @ t_2_skew)))
    C52 = -Mass[2] * (B_3_T @ (R_IP @ (omega_skew @ t_3_skew)))
    C33 = Mass[0] * (B_1_T @ B_1_dot)
    C44 = Mass[1] * (B_2_T @ B_2_dot)
    C55 = Mass[2] * (B_3_T @ B_3_dot)

    C_right = np.block([
        [C12, C13, C14, C15],
        [C22, C23, C24, C25],
        [C32, C33, np.zeros((2, 2)), np.zeros((2, 2))],
        [C42, np.zeros((2, 2)), C44, np.zeros((2, 2))],
        [C52, np.zeros((2, 2)), np.zeros((2, 2)), C55]
    ])
    
    C = np.hstack([C_left, C_right])

    return C

def get_G(x):

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

    # Matrix A
    A = get_A(Tether, Mass)
    A_T = A.T

    # Matrix J_q
    J_q = get_J(Tether, Mass)
    g_I = np.array([0, 0, g]).reshape(3, 1)

    # Gravity vector G (12x1)
    G1 = np.zeros((6, 1))
    G2 = B_1_T @ (-(m_p + Mass[0])/3 * g_I)
    G3 = B_2_T @ (-(m_p + Mass[1])/3 * g_I)
    G4 = B_3_T @ (-(m_p + Mass[2])/3 * g_I)
    G = np.vstack([G1, G2, G3, G4])

    return G