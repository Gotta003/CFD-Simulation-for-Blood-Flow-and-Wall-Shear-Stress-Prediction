# from src.models.utils import *
from src.models  import utils
import torch.nn as nn
import torch.nn.functional as F
import torch



class PinnLoss(nn.Module):
    def __init__(self, net, lambda_ = 0.9):
        super(PinnLoss, self).__init__()
        self.adaptive_constant_bc=0
        self.adaptive_constant_data=0
        self.net = net
        self.epoch = 0
        self.already_updated = False
        self.lambda_ = lambda_
        self.weight_loss_pinn = 0.3
        self.weight_loss_pred = 0.7
        self.diff = 0.04 # viscosity
        self.rho = 1.06 # density

    def criterion(self, points,  h_in, cfd):
        # MSE LOSS
        loss_f = nn.MSELoss()
        loss = 0
        
    

        for i in range(h_in.shape[3]):
            u = h_in[:, 1, :, i]
            v = h_in[:, 2, :, i ]
            w = h_in[:, 3, :, i]
            P = h_in[:, 0, :, i]
            # prepare the gradient to make the equations as losses
            grads_u = torch.autograd.grad(
                outputs=u, 
                inputs=points, 
                grad_outputs=torch.ones_like(u), 
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            u_x = grads_u[:, 0, :, i]
            grads_u_x = torch.autograd.grad(
                    outputs=u_x, 
                    inputs=points, 
                    grad_outputs=torch.ones_like(u_x), 
                    create_graph=True,
                    only_inputs=True
                )[0]
            u_xx = grads_u_x[:, 0, :, i]
            u_y = grads_u[:, 1, :, i]
            grads_u_y = torch.autograd.grad(
                    outputs=u_y, 
                    inputs=points, 
                    grad_outputs=torch.ones_like(u_y), 
                    create_graph=True,
                    only_inputs=True
                )[0]
            u_yy = grads_u_y[:, 1, :, i]            
            u_z = grads_u[:, 2, :, i]
            grads_u_z = torch.autograd.grad(
                    outputs=u_z, 
                    inputs=points, 
                    grad_outputs=torch.ones_like(u_z), 
                    create_graph=True,
                    only_inputs=True
                )[0]
            u_zz = grads_u_z[:, 2, :, i]
            u_T = grads_u[:, 3, :, i]
            
            grads_v = torch.autograd.grad(
                outputs=v, 
                inputs=points, 
                grad_outputs=torch.ones_like(u), 
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            v_x = grads_v[:, 0, :, i]
            grads_v_x = torch.autograd.grad(
                    outputs=v_x, 
                    inputs=points, 
                    grad_outputs=torch.ones_like(v_x), 
                    create_graph=True,
                    only_inputs=True
                )[0]
            v_xx = grads_v_x[:, 0, :, i]
            v_y = grads_v[:, 1, :, i]
            grads_v_y = torch.autograd.grad(
                    outputs=v_y, 
                    inputs=points, 
                    grad_outputs=torch.ones_like(v_y), 
                    create_graph=True,
                    only_inputs=True
                )[0]
            v_yy = grads_v_y[:, 1, :, i]
            v_z = grads_v[:, 2, :, i]
            grads_v_z = torch.autograd.grad(
                    outputs=v_z, 
                    inputs=points, 
                    grad_outputs=torch.ones_like(v_z), 
                    create_graph=True,
                    only_inputs=True
                )[0]
            v_zz = grads_v_z[:, 2, :, i]
            v_T = grads_v[:, 3, :, i]
            
            grads_w = torch.autograd.grad(
                outputs=w, 
                inputs=points, 
                grad_outputs=torch.ones_like(w), 
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            w_x = grads_w[:, 0, :, i]
            grads_w_x = torch.autograd.grad(
                    outputs=w_x, 
                    inputs=points, 
                    grad_outputs=torch.ones_like(w_x), 
                    create_graph=True,
                    only_inputs=True
                )[0]
            w_xx = grads_w_x[:, 0, :, i]
            w_y = grads_w[:, 1, :, i]
            grads_w_y = torch.autograd.grad(
                    outputs=w_y, 
                    inputs=points, 
                    grad_outputs=torch.ones_like(w_y), 
                    create_graph=True,
                    only_inputs=True
                )[0]
            w_yy = grads_w_y[:, 1, :, i]
            w_z = grads_w[:, 2, :, i]
            grads_w_z = torch.autograd.grad(
                    outputs=w_z, 
                    inputs=points, 
                    grad_outputs=torch.ones_like(w_z), 
                    create_graph=True,
                    only_inputs=True
                )[0]
            w_zz = grads_w_z[:, 2, :, i]
            w_T = grads_w[:, 3, :, i]
            grads_p = torch.autograd.grad(
                outputs=P, 
                inputs=points, 
                grad_outputs=torch.ones_like(P), 
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            P_x = grads_p[:, 0, :, i]
            P_y = grads_p[:, 1, :, i]
            P_z = grads_p[:, 2, :, i]

            
            # scale to magnitude the losses ( not necessary )
            loss_1 = u*u_x + v*u_y + w*u_z - (self.diff/self.rho)*(u_xx + u_yy + u_zz) + 1/self.rho * (P_x)  #X-dir
            loss_2 = u*v_x + v*v_y + w*v_z - (self.diff/self.rho)*(v_xx + v_yy + v_zz) + 1/self.rho * (P_y)  #Y-dir
            loss_3 = u*w_x + v*w_y + w*w_z - (self.diff/self.rho)*(w_xx + w_yy + w_zz) + 1/self.rho * (P_z)  #z-dir
            loss_4 = (u_x + v_y + w_z)   #continuity
            loss_1 = u_T + u*u_x + v*u_y + w*u_z - (self.diff/self.rho)*(u_xx + u_yy + u_zz) + 1/self.rho * (P_x)  #X-dir
            loss_2 = v_T + u*v_x + v*v_y + w*v_z - (self.diff/self.rho)*(v_xx + v_yy + v_zz) + 1/self.rho * (P_y)  #Y-dir
            loss_3 = w_T + u*w_x + v*w_y + w*w_z - (self.diff/self.rho)*(w_xx + w_yy + w_zz) + 1/self.rho * (P_z)  #z-dir
            loss_4 = (u_x + v_y + w_z) #continuity
            
            #Note our target is zero. It is residual so we use zeros_like
            loss = (loss 
                    + loss_f(loss_1, torch.zeros_like(loss_1)) 
                    + loss_f(loss_2, torch.zeros_like(loss_2)) 
                    + loss_f(loss_4, torch.zeros_like(loss_4)) 
                    + loss_f(loss_3, torch.zeros_like(loss_3)))
            
        return loss

    def Loss_BC(self, h_wall):
        out1_u = h_wall[:, 1, :]
        out1_v = h_wall[:, 2, :]
        out1_w = h_wall[:, 3, :]
        loss_f = nn.MSELoss()
        loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v)) + loss_f(out1_w, torch.zeros_like(out1_w))
        return loss_noslip

    def Loss_data(self, h_in, cfd):
        mse = nn.MSELoss()
        loss_d = 0
        for i in range(h_in.shape[3]):
            out1_u = h_in[:, 1, :, i]
            out1_v = h_in[:, 2, :, i]
            out1_w = h_in[:, 3, :, i]
            out1_wss = h_in[:, 4, :, i]
            ud = cfd[:, 1, :, i]
            vd = cfd[:, 2, :, i]
            wd = cfd[:, 3, :, i]
            wss = cfd[:, 4, :, i]
            loss_d = loss_d +  mse(out1_u, ud) + mse(out1_v, vd) + mse(out1_w, wd) + mse(out1_wss, wss)
        return loss_d

    def update_adaptive_constants(self, loss_eqn, loss_bc, loss_data):
        eq_max_grad = []
        bc_mean_grad = []
        data_mean_grad = []
        print("Updating adaptive constants...")
        # 1. Identifica tutti i pesi (escludendo i bias e altri parametri)
        # Usiamo una lista di tensori per iterare direttamente sugli oggetti
        target_weights = [p for n, p in self.net.named_parameters() if 'weight' in n]
        # 2. Itera direttamente sui tensori dei pesi
        for weight_tensor in target_weights:
            # Calcolo per loss_eqn (Max del gradiente)
            grad_eqn = torch.autograd.grad(loss_eqn, weight_tensor, create_graph=True, only_inputs=True, allow_unused=True)[0]
            if grad_eqn is not None:
                a = torch.max(torch.abs(grad_eqn)).cpu().detach().numpy()
                eq_max_grad.append(a)
            # Calcolo per loss_bc (Media del gradiente)
            grad_bc = torch.autograd.grad(loss_bc, weight_tensor, create_graph=True, only_inputs=True, allow_unused=True)[0]
            if grad_bc is not None:
                b = torch.mean(torch.abs(grad_bc)).cpu().detach().numpy()
                bc_mean_grad.append(b) 
            # Calcolo per loss_data (Media del gradiente)
            grad_data = torch.autograd.grad(loss_data, weight_tensor, create_graph=True, only_inputs=True, allow_unused=True)[0]
            if grad_data is not None:
                c = torch.mean(torch.abs(grad_data)).cpu().detach().numpy()
                data_mean_grad.append(c)
        maximum_grad_eq=max(eq_max_grad)
        mean_grad_bc= np.mean(bc_mean_grad)
        mean_grad_data=np.mean(data_mean_grad)
        if self.adaptive_constant_bc > 0:
            self.adaptive_constant_bc = (1-self.lambda_)*(maximum_grad_eq/mean_grad_bc) + self.lambda_*self.adaptive_constant_bc
            self.adaptive_constant_data = (1-self.lambda_)*(maximum_grad_eq/mean_grad_data) + self.lambda_*self.adaptive_constant_data
        else:
            self.adaptive_constant_bc = maximum_grad_eq/mean_grad_bc
            self.adaptive_constant_data = maximum_grad_eq/mean_grad_data

    def update_epoch(self):
        self.epoch += 1
        self.already_updated = False

    def forward(self, points, pred, h_in,  target, h_wall, cfd, cfd_wall):
        loss_data = self.Loss_data(h_in, cfd) 
        loss_eqn = self.criterion(points, h_in, cfd)
        loss_bc = self.Loss_BC(h_wall) 
        loss_pinn = loss_eqn + self.adaptive_constant_bc * loss_bc + self.adaptive_constant_data * loss_data
        loss_pred = nn.BCELoss()(pred, target)
        if ((self.epoch % 10 == 0) and not self.already_updated):
            self.update_adaptive_constants(loss_eqn, loss_bc, loss_data)
            self.already_updated = True
        return self.weight_loss_pinn * loss_pinn + self.weight_loss_pred * loss_pred 

class PointRNN(nn.Module):
    def __init__(self, size = 1088, radius = 0.2, nsample = 32):
        super(PointRNN, self).__init__()
        self.size = size
        self.radius = radius
        self.nsample = nsample
        self.linear = nn.Linear(2*self.size+6, self.size)
        
    def forward(self, p, x_t = None, h_t_minus_1 = None):
        if h_t_minus_1 is None:
            h_t_minus_1 = x_t
        neigh = query_ball_point(self.radius, self.nsample, p.permute(0, 2, 1), p.permute(0, 2, 1))
        neigh_pos = index_points(p.permute(0, 2, 1), neigh)
        dist_diff = neigh_pos - p.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, self.nsample, -1)
        neigh_plus_feat = index_points(torch.cat([p.permute(0, 2, 1), x_t.permute(0, 2, 1), h_t_minus_1.permute(0, 2, 1)], dim=-1), neigh)
        neigh_plus_feat = torch.cat([dist_diff, neigh_plus_feat], dim=-1)
        out, _ = torch.max(neigh_plus_feat, dim=2)
        out = self.linear(out)
        return out
    
class PointGRU(nn.Module):
    def __init__(self, size = 1088, radius = 0.2, nsample = 32):
        super(PointGRU, self).__init__()
        self.z_rnn = PointRNN(size, radius, nsample)
        self.r_rnn = PointRNN(size, radius, nsample)
        self.s_tilde_t_minus_1_rnn = PointRNN(size, radius, nsample)
        self.s_tilde_linear = nn.Linear(size+3, size)
        

    def forward(self, p, x_t, h_t_minus_1):
        z_t = torch.sigmoid(self.z_rnn(p, x_t, h_t_minus_1))
        r_t = torch.sigmoid(self.r_rnn(p, x_t, h_t_minus_1))
        s_tilde_t_minus_1 = self.s_tilde_t_minus_1_rnn(p, torch.zeros_like(x_t), h_t_minus_1)
        
        cat_input = torch.cat([p, x_t * s_tilde_t_minus_1.permute(0, 2, 1)], dim=1)
        
        s_tilde_t = torch.tanh(self.s_tilde_linear(cat_input.permute(0, 2, 1)))

        h_t = z_t * s_tilde_t_minus_1 + (torch.ones_like(z_t) - z_t) * s_tilde_t

 
        return h_t.permute(0, 2, 1)

class GNNPinn(nn.Module):
    def __init__(self, num_class=1,  normal_channel=3, feat_channel=97, num_other_patologies=11):
        super(GNNPinn, self).__init__()

        # recursive pinn
        self.encoder = PointNetEncoder(global_feat=False, feature_transform=True, channel=4)
        self.first_conv = nn.Conv1d(1088, 512, kernel_size=1)
        self.second_conv = nn.Conv1d(512, 64, kernel_size=1)
        self.point_gru = PointGRU(size=64, radius=0.2, nsample=32)
        self.out_encoder_conv = nn.Conv1d(64, 5, kernel_size=1)
        self.classifier_conv = nn.Conv1d(5, 3, kernel_size=1)
        # pointnet
        in_channel = normal_channel
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.sa4 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa5 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa6 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.sa7 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa8 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa9 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024*3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        self.fc4 = nn.Linear(256, num_other_patologies)
        self.mlp = nn.Sequential(
            nn.Linear(feat_channel, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
        )
        self.fusion = nn.Linear(512 + 1024*3, 1024*3)

    def forward(self, xyz, feat, xyz_wall):
        pred = None
        h = None
        h_list = []
        pinn_out_list = []
        for i in range(xyz.shape[3]):
            x, trans, trans_feat = self.encoder(xyz[:, :, :, i])
            x = self.first_conv(x)
            x = F.relu(x)
            x = self.second_conv(x)
            coord = xyz[:, :3, :, i] 
            h_t = self.point_gru(coord, x, h_list[-1] if h_list else None)
            pinn_out = self.out_encoder_conv(h_t)
            h_list.append(h_t)
            pinn_out_list.append(pinn_out)
        h = torch.stack(pinn_out_list, dim=1).permute(0, 2, 3, 1)
        # compute for the wall
        x_wall = torch.cat((xyz_wall, xyz[:, -1, :, -1].unsqueeze(1)), dim=1)
        x_wall, trans, trans_feat = self.encoder(x_wall)
        x_wall = self.first_conv(x_wall)
        x_wall = F.relu(x_wall)
        x_wall = self.second_conv(x_wall)
        h_wall = self.point_gru(xyz_wall, x_wall, h_list[-1])
        pinn_out_wall = self.out_encoder_conv(h_wall)
        last_pred = h[:, :, :, -1]
        last_pred = self.classifier_conv(last_pred)
        last_pred__wall = pinn_out_wall[:, :, :]       
        last_pred_wall = self.classifier_conv(last_pred__wall)
        pred = torch.cat([xyz[:, :3, :, -1], last_pred], dim=1)
        pred_wall = torch.cat([xyz_wall[:, :3, :], last_pred_wall], dim=1)
        pred = torch.cat([pred, pred_wall], dim=2)        
        B, _, _ = pred.shape
        norm1 = pred[:, 3:4, :]
        norm2 = pred[:, 4:5, :]
        norm3 = pred[:, 5:6, :]
        norm1 = norm1.repeat(1, 3, 1)
        norm2 = norm2.repeat(1, 3, 1)
        norm3 = norm3.repeat(1, 3, 1)
        pred = pred[:, :3, :]
        l1_xyz1, l1_points1 = self.sa1(pred, norm1)
        l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)
        l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)
        x1 = l3_points1.view(B, 1024)
        l1_xyz2, l1_points2 = self.sa4(pred, norm2)
        l2_xyz2, l2_points2 = self.sa5(l1_xyz2, l1_points2)
        l3_xyz2, l3_points2 = self.sa6(l2_xyz2, l2_points2)
        x2 = l3_points2.view(B, 1024)
        l1_xyz3, l1_points3 = self.sa7(pred, norm3)
        l2_xyz3, l2_points3 = self.sa8(l1_xyz3, l1_points3)
        l3_xyz3, l3_points3 = self.sa9(l2_xyz3, l2_points3)
        x3 = l3_points3.view(B, 1024)
        pred = torch.cat([x1, x2, x3], dim=1)  
        feat = self.mlp(feat)  
        pred = torch.cat([feat, pred], dim=-1) 
        pred = self.fusion(pred)  
        pred = self.drop1(F.relu(self.bn1(self.fc1(pred))))
        pred = self.drop2(F.relu(self.bn2(self.fc2(pred))))
        pred_1 = self.fc3(pred)
        pred_1 = torch.sigmoid(pred_1)
        return pred_1, h, pinn_out_wall, pred
    

