# from src.models.utils import *
from src.models.utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch



class Fakemodel(nn.Module):
    def __init__(self):
        super(Fakemodel, self).__init__()
        self.fc1 = nn.Linear(1024*3, 1024)
    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        x = torch.zeros(B, 1, requires_grad=True).to(xyz.device)
        return x
    
class PointNet(nn.Module):
    def __init__(self, num_class=1,  normal_channel=3, feat_channel=97):
        super(PointNet, self).__init__()
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

        self.mlp = nn.Sequential(
            nn.Linear(feat_channel, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
        )
        self.fusion = nn.Linear(512 + 1024*3, 1024*3)


    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        norm1 = xyz[:, 3:4, :]
        norm2 = xyz[:, 4:5, :]
        norm3 = xyz[:, 5:6, :]
        norm1 = norm1.repeat(1, 3, 1)
        norm2 = norm2.repeat(1, 3, 1)
        norm3 = norm3.repeat(1, 3, 1)
        xyz = xyz[:, :3, :]

        l1_xyz1, l1_points1 = self.sa1(xyz, norm1)
        l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)
        l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)
        x1 = l3_points1.view(B, 1024)
        l1_xyz2, l1_points2 = self.sa4(xyz, norm2)
        l2_xyz2, l2_points2 = self.sa5(l1_xyz2, l1_points2)
        l3_xyz2, l3_points2 = self.sa6(l2_xyz2, l2_points2)
        x2 = l3_points2.view(B, 1024)
        l1_xyz3, l1_points3 = self.sa7(xyz, norm3)
        l2_xyz3, l2_points3 = self.sa8(l1_xyz3, l1_points3)
        l3_xyz3, l3_points3 = self.sa9(l2_xyz3, l2_points3)
        x3 = l3_points3.view(B, 1024)


        x = torch.cat([x1, x2, x3], dim=1)  


        feat = self.mlp(feat)  
        x = torch.cat([feat, x], dim=-1) 
        x = self.fusion(x)  


        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        pred = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(pred)
        x = torch.sigmoid(x)
        return x, pred


