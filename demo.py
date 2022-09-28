import torch
import os
import os.path as osp
import numpy as np
import open3d as o3d

from arguments import Arguments
from models.scarp import SCARP

def orthonormalize_basis(basis):
    """
    Returns orthonormal basis vectors
    basis - B, 3, 3

    out - B, 3, 3
    """
    u, s, v = torch.svd(basis)
    out = u @ v.transpose(-2, -1)    

    return out

def return_rotated_pcd(basis, shape_comp):
        """
        Returns the pointcloud in the original pose by

        basis: B x 3 x 3
        shape_comp: B x N x 3

        returns: B x N x 3
        """
        basis = torch.stack(basis, dim = 1)

        orth_basis = orthonormalize_basis(basis)
        orth_basis = orth_basis[0]
        y_p = torch.einsum("bij, bpj->bpi", orth_basis, shape_comp)
        y_p = torch.stack([y_p[..., 2], y_p[..., 0], y_p[..., 1]], dim = -1)

        return y_p

if __name__ == '__main__':

    # load arguments
    args = Arguments(stage='demo').parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    os.makedirs('./checkpoints',exist_ok=True)
    load_ckpt = args.ckpt_load if args.ckpt_load is not None else None
    assert load_ckpt != None, print('Invalid Checkpoint Path. Aborting.')
    print(f"checkpoint is :{load_ckpt}")
    
    # load model
    model = SCARP(args)
    checkpoint = torch.load(load_ckpt, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded.")
    model = model.to(args.device)
    model.eval()

    # load input path
    pcd_path = os.path.join(args.demo_dataset_path,args.class_choice)

    with torch.no_grad():
        for path in os.listdir(pcd_path):
            if path.endswith('_output.pcd') or not path.endswith('_input.pcd'):
                continue

            # load input pointcloud
            f = os.path.join(pcd_path,path)
            pcd = o3d.io.read_point_cloud(f)
            input_point_cloud = np.asarray(pcd.points).astype(np.float32)
            input_point_cloud = torch.from_numpy(input_point_cloud).unsqueeze(dim=0).to(args.device)
            
            # forward pass
            output = model(input_point_cloud)

            # get the pointcloud in the orginial pose
            basis,canonical_pcd,translation = output['E'],output['pcd'], - output['T'][0]
            rotated_pcd = return_rotated_pcd(basis,canonical_pcd)
            output_pcd = rotated_pcd + translation

            # save pointcloud
            save_path = path.split('_')[0] + '_output.pcd'
            save_path = os.path.join(pcd_path,save_path)
            pcd.points = o3d.utility.Vector3dVector(output_pcd.squeeze(dim=0).cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pcd.points) * [0.5, 0.5, 0.5])
            o3d.io.write_point_cloud(save_path,pcd)

