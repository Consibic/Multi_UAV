# get Z, n, B, Bdot
import torch

def get_R(phi, theta, psi):
    R11 = torch.cos(theta) * torch.cos(psi)
    R12 = -torch.cos(theta) * torch.sin(psi)
    R13 = torch.sin(theta)
    R21 = torch.cos(phi) * torch.sin(psi) + torch.sin(phi) * torch.sin(theta) * torch.cos(psi)
    R22 = torch.cos(phi) * torch.cos(psi) - torch.sin(phi) * torch.sin(theta) * torch.sin(psi)
    R23 = -torch.sin(phi) * torch.cos(theta)
    R31 = torch.sin(phi) * torch.sin(psi) - torch.cos(phi) * torch.sin(theta) * torch.cos(psi)
    R32 = torch.sin(phi) * torch.cos(psi) + torch.cos(phi) * torch.sin(theta) * torch.sin(psi)
    R33 = torch.cos(phi) * torch.cos(theta)
    return torch.tensor([[R11, R12, R13],
                         [R21, R22, R23], 
                         [R31, R32, R33]])


def get_Z(r, l):
    Z = torch.sqrt(torch.abs(torch.tensor(l)**2 - (r[0][0][0]**2 + r[0][1][0]**2)))
    return Z


def get_B(r, l):
    bs = r.shape[0]
    Z = get_Z(r, l)
    I2 = torch.eye(2).repeat(bs, 1, 1).type(r.type())  # shape (bs, 2, 2)
    B = torch.cat([I2, r.transpose(1, 2)/Z], dim=1)
    return B


def get_B_dot(r, v, l):
    bs = r.shape[0]
    O2 = torch.zeros(bs, 2, 2).type(r.type())  # shape (bs, 2, 2)
    Z = get_Z(r, l)
    r_T = r.transpose(1, 2)  # shape (bs, 1, 2)
    v_T = v.transpose(1, 2)  # shape (bs, 1, 2)
    B_dot = torch.cat([O2, ((Z) ** 2 * v_T + torch.matmul(r_T, v) * r_T)/((Z) ** 3)], dim=1)  # shape (bs, 3, 2)
    return B_dot


def skew_symmetric(v):
    return torch.tensor([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]], 
                         [-v[1], v[0], 0]], dtype=float)


def get_A(l, m):
    A = torch.zeros((1, 3, 3))
    for j in range(len(m)):
        t_j_skew = skew_symmetric(l[:, j])
        A += t_j_skew * m[j]
    return A


def get_J(l, m):
    J = torch.zeros((3, 3))
    for j in range(len(m)):
        t_j_skew = skew_symmetric(l[:, j])
        J += -m[j] * torch.mm(t_j_skew, t_j_skew)
    return J