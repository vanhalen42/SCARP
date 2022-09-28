import torch
import torch.nn as nn
import torch.nn.functional as F

from spherical_harmonics.spherical_cnn import torch_fibonnacci_sphere_sampling, SphericalHarmonicsEval, SphericalHarmonicsCoeffs, zernike_monoms

from models.layers import MLP, MLP_layer, set_sphere_weights, apply_layers, type_1
from models.treeGAN.treegan_network import Generator
from models.TFN import TFN
from models.pointnet2.pointnet import Pointnet

class SCARP(nn.Module):
    '''
    Module for SCARP
    '''
    def __init__(self,args=None):
        super(SCARP,self).__init__()

        self.args = args

        #Model
        self.pointnet2 = Pointnet(0)

        num_frames = 1
        self.bn_momentum = 0.75
        self.basis_dim = 3
        self.l_max = [3, 3, 3]
        self.l_max_out = [3, 3, 3]
        self.num_shells = [3, 3, 3]
        self.num_frames = num_frames

        sphere_samples = 64
        self.basis_dim = 3

        self.mlp_units = [[32, 32], [64, 64], [128, 256]]

        self.tfn = TFN(sphere_samples = sphere_samples,
            bn_momentum = 0.75, 
            mlp_units = [[32, 32], [64, 64], [128, 256]], 
            l_max = [3, 3, 3], l_max_out = [3, 3, 3], 
            num_shells = [3, 3, 3])

        self.basis_mlp = []
        self.basis_layer = []

        self.basis_units = [64]
        for frame_num in range(num_frames):
            self.basis_mlp.append(MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.basis_units, bn_momentum = 0.75))
            self.basis_layer.append(MLP(in_channels = self.basis_units[-1], out_channels=self.basis_dim, apply_norm = False))

        self.basis_mlp = torch.nn.Sequential(*self.basis_mlp)
        self.basis_layer = torch.nn.Sequential(*self.basis_layer)

        self.reduction_dim = 128
        self.reduction_layer_params = [128]

        self.S2 = torch_fibonnacci_sphere_sampling(sphere_samples)

        self.translation_mlp = (MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.basis_units, bn_momentum = self.bn_momentum))
        self.translation_layer = MLP(in_channels = self.basis_units[-1], out_channels=1, apply_norm = False)

        self.type_1                     = SphericalHarmonicsCoeffs(l_list=[1], base=self.S2)
        
        self.reduction_mlp = MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.reduction_layer_params, bn_momentum = 0.75)
        self.reduction_layer = MLP(in_channels = self.reduction_layer_params[-1], out_channels=self.reduction_dim, apply_norm = False)

        self.TreeGan = Generator(features=args.G_FEAT, degrees=args.DEGREE, support=args.support,args=self.args)

    def forward(self, gt=None, partial=None, infer=False):
        pointnet2_features = self.pointnet2(gt.permute(0,2,1)) # B x 128

        tfn_raw_features, tfn_features = self.tfn(gt) # B x 64 x 256
        tfn_global_invarient_features = torch.max(tfn_features, dim=1)[0] # B x 128

        E = []
        T = []
        F_translation = tfn_raw_features

        # Compute equivariant layers
        for frame_num in range(self.num_frames):
            basis = self.basis_mlp[frame_num](tfn_raw_features)
            basis = self.basis_layer[frame_num](basis)
            basis = self.type_1.compute(basis)['1']
            basis = torch.nn.functional.normalize(basis, dim=-1, p = 2, eps = 1e-6)
            E.append(basis)
        
        # Predicting translation
        translation = self.translation_mlp(F_translation)
        y_dict = self.type_1.compute(translation)
        translation = y_dict['1'] 
        translation = self.translation_layer(translation)
        translation = torch.stack([translation[:, 2], translation[:, 0], translation[:, 1]], dim = -1)
        T.append(translation)
        
        # Combining TFN globally invarian features with pointnet++ features
        concatenated_features = torch.cat((pointnet2_features, tfn_global_invarient_features), dim=1) # B x 256
        reduced_features = self.reduction_mlp(concatenated_features) # B x 128
        reduced_features = self.reduction_layer(reduced_features) # B x 128

        return { "pcd": self.TreeGan([reduced_features.unsqueeze(1)]), "E": E,"T":T}


if __name__ == '__main__':
    x = torch.rand((32,2048,3))

    model = SCARP()
    model(x)
