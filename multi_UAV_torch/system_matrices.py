# System matrices for the slung payload system with variable length using Kane's method.
import torch
import numpy as np
from .kinematics import get_A, get_J, get_Z, get_R, get_B, get_B_dot, skew_symmetric


class SlungPayloadVariableLengthSystem:
    def __init__(self, x_init):
        # Training parameters
        self.type = x_init.type()
        self.bs = x_init.shape[0]
        self.num_dim_x = 24  # number of dimensions in state vector x
        self.num_dim_control = 9  # number of dimensions in control vector u
        
        # Parameters
        self.g = 9.81
        self.m_p = 1.3  # mass of the payload
        self.m_q = np.array([1.63, 1.63, 1.63])  # mass of the quadrotor
        self.J_p = np.diag([0.5, 0.5, 0.5])
        self.l = np.array([0.98, 0.98, 0.98])  # cable length
        self.t = np.array([[1.085, 0, 0],  # cable anchor points
                   [-0.5425, -0.9396, 0],
                   [-0.5425, 0.9396, 0]])
        
        # States
        self.x_p = torch.zeros(self.bs, 3, 1).type(x_init.type())  # (bs, 3, 1)
        self.r_j = torch.zeros(self.bs, 6, 1).type(x_init.type())  # (bs, 6, 1)
        self.v_p = torch.zeros(self.bs, 3, 1).type(x_init.type())  # (bs, 3, 1)
        self.omega_p = torch.zeros(self.bs, 3, 1).type(x_init.type())  # (bs, 3, 1)
        self.v_q = torch.zeros(self.bs, 6, 1).type(x_init.type())  # (bs, 6, 1)
        self.l_dot = torch.zeros(self.bs, 1, 1).type(x_init.type())  # (bs, 1, 1)
        self.R = torch.zeros(self.bs, 3, 3).type(x_init.type())  # (bs, 3, 3)
        
        # Kinematics
        self.Z_1 = get_Z(self.r_j[:, 0:2, :], self.l[0])
        self.Z_2 = get_Z(self.r_j[:, 2:4, :], self.l[1])
        self.Z_3 = get_Z(self.r_j[:, 4:6, :], self.l[2])
        self.B_1 = get_B(self.r_j[:, 0:2, :], self.l[0])
        self.B_2 = get_B(self.r_j[:, 2:4, :], self.l[1])
        self.B_3 = get_B(self.r_j[:, 4:6, :], self.l[2])
        self.B_1_T = self.B_1.transpose(1, 2)
        self.B_2_T = self.B_2.transpose(1, 2)
        self.B_3_T = self.B_3.transpose(1, 2)
        self.B_1_dot = get_B_dot(self.r_j[:, 0:2, :], self.v_q[:, 0:2, :], self.l[0])
        self.B_2_dot = get_B_dot(self.r_j[:, 2:4, :], self.v_q[:, 2:4, :], self.l[1])
        self.B_3_dot = get_B_dot(self.r_j[:, 4:6, :], self.v_q[:, 4:6, :], self.l[2])
        self.J_q = get_J(self.t, self.m_q)
        self.A = get_A(self.t, self.m_q)
        self.A_T = self.A.transpose(1, 2)[0]
        
        # MCG matrices
        self.M = torch.zeros(self.bs, 12, 12).type(x_init.type())
        self.C = torch.zeros(self.bs, 12, 12).type(x_init.type())
        self.F_g = torch.zeros(self.bs, 12, 1).type(x_init.type())
        
        # fx, gx
        self.fx = torch.zeros(self.bs, self.num_dim_x, 1).type(self.type)
        self.gx = torch.zeros(self.bs, self.num_dim_x, self.num_dim_control).type(self.type)
        

    def update_state(self, x):
        # Training parameters
        self.type = x.type()
        self.bs = x.shape[0]
        self.num_dim_x = 24
        self.num_dim_control = 4
         
        # States
        x_p_x, x_p_y, x_p_z, phi, theta, psi, r_1_x, r_1_y, r_2_x, r_2_y, r_3_x, r_3_y, v_p_x, v_p_y, v_p_z, omega_phi, omega_theta, omega_psi, v_q1_x, v_q1_y, v_q2_x, v_q2_y, v_q3_x, v_q3_y = [x[:, i, 0] for i in range(self.num_dim_x)]
        self.x_p = torch.stack([x_p_x, x_p_y, x_p_z], dim=1).unsqueeze(-1)
        self.r_j = torch.stack([r_1_x, r_1_y, r_2_x, r_2_y, r_3_x, r_3_y], dim=1).unsqueeze(-1)
        self.v_p = torch.stack([v_p_x, v_p_y, v_p_z], dim=1).unsqueeze(-1)
        self.omega_p = torch.stack([omega_phi, omega_theta, omega_psi], dim=1).unsqueeze(-1)
        self.v_q = torch.stack([v_q1_x, v_q1_y, v_q2_x, v_q2_y, v_q3_x, v_q3_y], dim=1).unsqueeze(-1)
        # self.l_dot = l_dot.view(-1, 1, 1)
        self.R = get_R(phi, theta, psi)
        
        # Kinematics 
        self.Z_1 = get_Z(self.r_j[:, 0:2, :], self.l[0])
        self.Z_2 = get_Z(self.r_j[:, 2:4, :], self.l[1])
        self.Z_3 = get_Z(self.r_j[:, 4:6, :], self.l[2])
        self.B_1 = get_B(self.r_j[:, 0:2, :], self.l[0])
        self.B_2 = get_B(self.r_j[:, 2:4, :], self.l[1])
        self.B_3 = get_B(self.r_j[:, 4:6, :], self.l[2])
        self.B_1_T = self.B_1.transpose(1, 2)
        self.B_2_T = self.B_2.transpose(1, 2)
        self.B_3_T = self.B_3.transpose(1, 2)
        self.B_1_dot = get_B_dot(self.r_j[:, 0:2, :], self.v_q[:, 0:2, :], self.l[0])
        self.B_2_dot = get_B_dot(self.r_j[:, 2:4, :], self.v_q[:, 2:4, :], self.l[1])
        self.B_3_dot = get_B_dot(self.r_j[:, 4:6, :], self.v_q[:, 4:6, :], self.l[2])

        # Skew metrics
        self.t_1_skew = skew_symmetric(self.t[:, 0])
        self.t_2_skew = skew_symmetric(self.t[:, 1])
        self.t_3_skew = skew_symmetric(self.t[:, 2])
        
        # Update M, C, F_g
        self.calc_M()
        self.calc_C()
        self.calc_G()
        
        # Update fx, gx
        self.calc_fx()
        self.calc_gx()

    def get_M(self):
        return self.M
    
    def get_C(self):
        return self.C
    
    def get_G(self):
        return self.F_g
    
    def get_f(self):
        return self.fx

    def get_g(self):
        return self.gx

    def calc_M(self):
        # Mass matrix M (12x12)
        M11 = (self.m_p + np.sum(self.m_q)) * torch.eye(3)
        M12 = torch.mm(-self.R.float(), self.A_T)
        M13 = self.m_q[0] * self.B_1
        M14 = self.m_q[1] * self.B_2
        M15 = self.m_q[2] * self.B_3
        M22 = np.add(self.J_p, self.J_q.numpy())
        M23 = self.m_q[0] * torch.mm(self.t_1_skew, torch.mm(self.R, self.B_1[0]))
        M24 = self.m_q[1] * torch.mm(self.t_2_skew, torch.mm(self.R, self.B_2[0]))
        M25 = self.m_q[2] * torch.mm(self.t_3_skew, torch.mm(self.R, self.B_3[0]))
        M33 = self.m_q[0] * torch.mm(self.B_1_T[0], self.B_1[0])
        M44 = self.m_q[1] * torch.mm(self.B_2_T[0], self.B_2[0])
        M55 = self.m_q[2] * torch.mm(self.B_3_T[0], self.B_3[0])
        
        # Assemble the full mass matrix
        self.M = torch.cat([
            torch.cat([M11, M12, M13[0], M14[0], M15[0]], dim=1),
            torch.cat([M12.T, torch.Tensor(M22), M23, M24, M25], dim=1),
            torch.cat([M13[0].T, M23.T, M33, torch.zeros((2, 2)), torch.zeros((2, 2))], dim=1),
            torch.cat([M14[0].T, M24.T, torch.zeros((2, 2)), M44, torch.zeros((2, 2))], dim=1),
            torch.cat([M15[0].T, M25.T, torch.zeros((2, 2)), torch.zeros((2, 2)), M55], dim=1)
        ], dim=0)

    def calc_C(self):
        omega_p_skew = skew_symmetric(self.omega_p[0])
        # Compute C matrix
        C_left = torch.zeros(12, 3)
        C12 = torch.mm(self.R.float(), torch.mm(skew_symmetric(self.omega_p[0]).float(), self.A_T.float()).float())
        C13 = self.m_q[0] * self.B_1_dot
        C14 = self.m_q[1] * self.B_2_dot
        C15 = self.m_q[2] * self.B_3_dot
        C22 = -skew_symmetric(torch.mm(torch.Tensor(self.J_p), self.omega_p[0].float())) - ((torch.tensor(self.m_q[0]) * torch.mm(self.t_1_skew, torch.mm(omega_p_skew, self.t_1_skew))) +
                                            (torch.tensor(self.m_q[1]) * torch.mm(self.t_2_skew, torch.mm(omega_p_skew, self.t_2_skew))) +
                                            (torch.tensor(self.m_q[2]) * torch.mm(self.t_3_skew, torch.mm(omega_p_skew, self.t_3_skew)))
                                            )
        C23 = self.m_q[0] * torch.mm(self.t_1_skew, torch.mm(self.R, self.B_1_dot[0]))
        C24 = self.m_q[1] * torch.mm(self.t_2_skew, torch.mm(self.R, self.B_2_dot[0]))
        C25 = self.m_q[2] * torch.mm(self.t_3_skew, torch.mm(self.R, self.B_3_dot[0]))
        C32 = -self.m_q[0] * torch.mm(self.B_1_T[0], torch.mm(self.R, torch.mm(omega_p_skew, self.t_1_skew)))
        C42 = -self.m_q[1] * torch.mm(self.B_2_T[0], torch.mm(self.R, torch.mm(omega_p_skew, self.t_2_skew)))
        C52 = -self.m_q[2] * torch.mm(self.B_3_T[0], torch.mm(self.R, torch.mm(omega_p_skew, self.t_3_skew)))
        C33 = self.m_q[0] * torch.mm(self.B_1_T[0], self.B_1_dot[0])
        C44 = self.m_q[1] * torch.mm(self.B_2_T[0], self.B_2_dot[0])
        C55 = self.m_q[2] * torch.mm(self.B_3_T[0], self.B_3_dot[0])

        C_right = torch.cat([
            torch.cat([C12, C13[0], C14[0], C15[0]], dim=1),
            torch.cat([C22, C23, C24, C25], dim=1),
            torch.cat([C32, C33, torch.zeros(2, 2), torch.zeros(2, 2)], dim=1),
            torch.cat([C42, torch.zeros(2, 2), C44, torch.zeros(2, 2)], dim=1),
            torch.cat([C52, torch.zeros(2, 2), torch.zeros(2, 2), C55], dim=1)
        ], dim=0)
        
        self.C = torch.cat([C_left, C_right], dim=1)
    
    def calc_G(self):
        # Compute G matrix
        g_I = torch.tensor([0, 0, -self.g]).view(1, 3, 1).expand(self.bs, 3, 1).type(self.type)  # (bs, 3, 1)

        # Gravity vector G (12x1)
        G1 = torch.zeros(6, 1)
        G2 = torch.mm(self.B_1_T[0], (-self.m_p/3 * g_I[0]))
        G3 = torch.mm(self.B_2_T[0], (-self.m_p/3 * g_I[0]))
        G4 = torch.mm(self.B_3_T[0], (-self.m_p/3 * g_I[0]))
        self.F_g = torch.cat([G1, G2, G3, G4], dim=0)

    def calc_fx(self):
        # Compute fx
        u = torch.cat([self.v_p, self.omega_p, self.v_q], dim=1)
        fx_upper = u[0]
        fx_lower = torch.linalg.solve(self.M, self.F_g - torch.mm(self.C, u[0]))
        self.fx = torch.cat([fx_upper, fx_lower], dim=1)

    def calc_gx(self):
        # Compute gx
        # Control input matrix H (12x9)
        H_lower = torch.block_diag(self.B_1_T[0], self.B_2_T[0], self.B_3_T[0])
        H = torch.cat([torch.zeros((6, 9)), H_lower], dim=0)

        gx_upper = torch.zeros((12, 9))
        gx_lower = torch.linalg.solve(self.M, H)

        self.gx = torch.cat([gx_upper, gx_lower], dim=1)
