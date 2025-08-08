import unittest
import numpy as np
import torch
import multi_UAV_np.system_matrices as ul
import multi_UAV_np.control_affine as ca
import multi_UAV_torch.system_matrices as torch_pkg
import multi_UAV_torch.kinematics as kine

class TestNumpyTorch(unittest.TestCase):
    def setUp(self):
        self.delta = 1e-5  # tolerance
        self.num_trials = 10
        self.l_bounds = (0.5, 2.0)  # bounds for l
        self.state_bounds = (-1.0, 1.0)  # bounds for other states
        self.m_p = 0.5
        self.m_q = 1.5
        self.g = 9.81

    def random_state(self):
        x = np.random.uniform(*self.state_bounds, size=24)
        x[5] = np.random.uniform(*self.l_bounds)
        # Ensure r1^2 + r2^2 < 1-eps for numerical safety
        while True:
            r = np.random.uniform(-0.99, 0.99, size=2)
            if np.sum(r**2) < 1 - 1e-6:
                break
        x[0:2] = r
        return x

    def test_np_torch(self):
        for trial in range(self.num_trials):
            with self.subTest(trial=trial):
                print(f"Running trial {trial+1}/{self.num_trials}")
                l = 0.98

                ######## NumPy Part ########

                # NumPy array
                self.x = self.random_state()
                self.r = self.x[6:8]
                self.v_p = self.x[12:14]
                self.l_dot = self.x[11]
                self.v_q = self.x[18:20]

                # NumPy kinematics
                z_comp = ul.get_z_comp(l, self.r)
                B = ul.get_B(l, self.r)
                B_dot = ul.get_B_dot(l, self.r, self.v_p)

                # NumPy system matrices
                M = ul.get_M(self.x)
                C = ul.get_C(self.x)
                G = ul.get_G(self.x)

                # NumPy control affine
                fx = ca.get_fx(self.x)
                gx = ca.get_gx(self.x)


                ######## Torch Part ########

                # Torch tensors
                x_torch = torch.tensor(self.x, dtype=torch.float64).view(1, 24, 1)
                r_torch = x_torch[:, 6:8, 0].unsqueeze(-1)
                v_p_torch = x_torch[:, 12:14, 0].unsqueeze(-1)
                v_q_torch = x_torch[:, 18:20, 0].unsqueeze(-1)

                # Torch kinematics
                z_comp_torch = kine.get_Z(r_torch, l)
                B_torch = kine.get_B(r_torch, l)
                B_dot_torch = kine.get_B_dot(r_torch, v_p_torch, l)
                z_comp_torch = z_comp_torch.squeeze().detach().cpu().numpy()
                B_torch = B_torch[0].detach().cpu().numpy()
                B_dot_torch = B_dot_torch[0].detach().cpu().numpy()

                # Torch system matrices
                system = torch_pkg.SlungPayloadVariableLengthSystem(x_torch)
                system.update_state(x_torch)
                M_torch = system.get_M()[0].detach().cpu().numpy()
                C_torch = system.get_C()[0].detach().cpu().numpy()
                G_torch = system.get_G()[0].detach().cpu().numpy()

                # Torch control affine
                fx_torch = system.get_f()[0].detach().cpu().numpy()
                gx_torch = system.get_g()[0].detach().cpu().numpy()

                print("-" * 20, "Kinematics Test", "-" * 20)
                self.assertAlmostEqual(z_comp, z_comp_torch, msg=f"z_comp failed", delta=self.delta)
                for i in range(B.shape[0]):
                    for j in range(B.shape[1]):
                        self.assertAlmostEqual(B[i, j], B_torch[i, j], msg=f"B failed at {i},{j}", delta=self.delta)
                for i in range(B_dot.shape[0]):
                    for j in range(B_dot.shape[1]):
                        self.assertAlmostEqual(B_dot[i, j], B_dot_torch[i, j], msg=f"B_dot failed at {i},{j}", delta=self.delta)
                print("-" * 20, "Kinematics Test: OK", "-" * 20)

                print("-" * 20, "System Matrices Test", "-" * 20)
                print("-------------------"
                      )
                print(M)
                print(M_torch)
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        self.assertAlmostEqual(M[i][0], M_torch[i], msg=f"M failed at {i},{j}", delta=self.delta)
                for i in range(C.shape[0]):
                    for j in range(C.shape[1]):
                        self.assertAlmostEqual(C[i][0], C_torch[i], msg=f"C failed at {i},{j}", delta=self.delta)
                for i in range(G.shape[0]):
                    for j in range(G.shape[1]):
                        self.assertAlmostEqual(G[i, j], G_torch[i], msg=f"G failed at {i},{j}", delta=self.delta)
                print("-" * 20, "System Matrices Test: OK", "-" * 20)

                print("-" * 20, "Control Affine Test", "-" * 20)
                for i in range(fx.shape[0]):
                    self.assertAlmostEqual(fx[i, 0], fx_torch[i, 0], msg=f"fx failed at {i}", delta=self.delta)
                for i in range(gx.shape[0]):
                    for j in range(gx.shape[1]):
                        self.assertAlmostEqual(gx[i, j], gx_torch[i, j], msg=f"gx failed at {i},{j}", delta=self.delta)
                print("-" * 20, "Control Affine Test: OK", "-" * 20)

                print(f"Trial {trial+1}/{self.num_trials} : OK.\n")

    def test_fxgx_torch_switched_state(self):
        """Test fxgx_torch functions with switched position states against control affine system"""
        for trial in range(self.num_trials):
            with self.subTest(trial=trial):
                print(f"Running fxgx_torch switched state trial {trial+1}/{self.num_trials}")

                ######## Generate original state ########
                x_original = self.random_state()
                # Original: [r_p_x, r_p_y, r_q_x, r_q_y, r_q_z, l, v_p_x, v_p_y, v_q_x, v_q_y, v_q_z, l_dot]
                
                # Create switched state for fxgx_torch (positions swapped)
                # Switched: [r_q_x, r_q_y, r_q_z, r_p_x, r_p_y, l, v_p_x, v_p_y, v_q_x, v_q_y, v_q_z, l_dot]
                x_switched = np.zeros_like(x_original)
                x_switched[0] = x_original[2]  # r_q_x -> position 0
                x_switched[1] = x_original[3]  # r_q_y -> position 1  
                x_switched[2] = x_original[4]  # r_q_z -> position 2
                x_switched[3] = x_original[0]  # r_p_x -> position 3
                x_switched[4] = x_original[1]  # r_p_y -> position 4
                x_switched[5:] = x_original[5:]  # rest unchanged

                ######## Control Affine System (original state) ########
                fx_control = ca.get_fx(x_original)
                gx_control = ca.get_gx(x_original)

                ######## fxgx_torch (switched state) ########
                x_torch_switched = torch.tensor(x_switched, dtype=torch.float64).view(1, 24, 1)
                sul = torch_pkg.SlungPayloadVariableLengthSystem(x_torch_switched)
                sul.update_state(x_torch_switched)
                fx_torch_raw = sul.get_f()[0].detach().cpu().numpy()
                gx_torch_raw = sul.get_g()[0].detach().cpu().numpy()

                # Reorder fx_torch back to original state ordering
                # fxgx_torch state: [r_q_x, r_q_y, r_q_z, r_p_x, r_p_y, l, v_p_x, v_p_y, v_q_x, v_q_y, v_q_z, l_dot]
                # Original state:   [r_p_x, r_p_y, r_q_x, r_q_y, r_q_z, l, v_p_x, v_p_y, v_q_x, v_q_y, v_q_z, l_dot]
                fx_torch_reordered = np.zeros_like(fx_torch_raw)
                fx_torch_reordered[0] = fx_torch_raw[3]  # r_p_x derivative
                fx_torch_reordered[1] = fx_torch_raw[4]  # r_p_y derivative  
                fx_torch_reordered[2] = fx_torch_raw[0]  # r_q_x derivative
                fx_torch_reordered[3] = fx_torch_raw[1]  # r_q_y derivative
                fx_torch_reordered[4] = fx_torch_raw[2]  # r_q_z derivative
                fx_torch_reordered[5:] = fx_torch_raw[5:]  # rest unchanged

                # Reorder gx_torch back to original state ordering
                gx_torch_reordered = np.zeros_like(gx_torch_raw)
                gx_torch_reordered[0] = gx_torch_raw[3]  # r_p_x row
                gx_torch_reordered[1] = gx_torch_raw[4]  # r_p_y row
                gx_torch_reordered[2] = gx_torch_raw[0]  # r_q_x row
                gx_torch_reordered[3] = gx_torch_raw[1]  # r_q_y row
                gx_torch_reordered[4] = gx_torch_raw[2]  # r_q_z row
                gx_torch_reordered[5:] = gx_torch_raw[5:]  # rest unchanged

                print("-" * 20, "fxgx_torch Switched State Test", "-" * 20)
                
                # Compare fx (should match after reordering)
                for i in range(fx_control.shape[0]):
                    self.assertAlmostEqual(fx_control[i, 0], fx_torch_reordered[i], 
                                         msg=f"fx_switched failed at {i}", delta=self.delta)
                
                # Compare gx (should match after reordering)
                for i in range(gx_control.shape[0]):
                    for j in range(gx_control.shape[1]):
                        self.assertAlmostEqual(gx_control[i, j], gx_torch_reordered[i, j], 
                                             msg=f"gx_switched failed at {i},{j}", delta=self.delta)
                
                print("-" * 20, "fxgx_torch Switched State Test: OK", "-" * 20)
                print(f"fxgx_torch switched trial {trial+1}/{self.num_trials} : OK.\n")


if __name__ == "__main__":
    unittest.main()