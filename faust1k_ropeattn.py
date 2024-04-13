import torch
from tqdm import tqdm
from argparse import ArgumentParser
from transmatching.Utils.utils import  get_errors, chamfer_loss, area_weighted_normalization, approximate_geodesic_distances
import numpy as np
from scipy.io import loadmat
from x_transformers import Encoder
import torch.nn as nn
import os
import random
import numpy
from point_gaussian import GaussianAttention

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    set_seed(0)


    faust = loadmat(args.path_data)
    shapes = faust["vertices"]
    faces = faust["f"] - 1
    n_pairs = 100
    n = shapes.shape[0]

    model = Encoder(
        dim=512,
        depth=6,
        heads=8,
        dim_head_custom = 64,
        pre_norm=False,
        residual_attn=True,
        rotary_pos_emb=True,
        rotary_emb_dim=64,
        attn_gaussian_heads=args.gaussian_heads,
        attn_force_gaussian_cross_attn=args.force_cross_attn,
        attn_legacy_force_cross_attn=args.legacy_force_cross_attn,
    ).to(args.device)

    gauss_attn = GaussianAttention(args.sigma).to(args.device)

    linear1 = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 64), nn.Tanh(),
                            nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 256), nn.Tanh(), nn.Linear(256, 512)).to(args.device)

    linear2 = nn.Sequential(nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 64),
                            nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(),
                            nn.Linear(16, 3)).to(args.device)


    modelname = args.run_name + ".pt"
    pathfolder= args.path_model
    model.load_state_dict(torch.load(os.path.join(pathfolder, modelname), map_location=lambda storage, loc: storage))
    linear1.load_state_dict(torch.load(os.path.join(pathfolder, "l1." + modelname), map_location=lambda storage, loc: storage))
    linear2.load_state_dict(torch.load(os.path.join(pathfolder, "l2." + modelname), map_location=lambda storage, loc: storage))
    gauss_attn.load_state_dict(torch.load(os.path.join(pathfolder, "gauss_attn." + modelname), map_location=lambda storage, loc: storage))

    print(modelname)
    print(model)
    print("MODEL RESUMED ---------------------------------------------------------------------------------------\n")

# ------------------------------------------------------------------------------------------------------------------
# END SETUP  -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
    model.eval()
    linear1.eval()
    linear2.eval()
    gauss_attn.eval()

    with torch.no_grad():
        err = []
        for _ in tqdm(range(n_pairs)):

            shape_A_idx = np.random.randint(n)
            shape_B_idx = np.random.randint(n)
            while shape_A_idx == shape_B_idx:  # avoid taking A exactly equal to B
                shape_B_idx = np.random.randint(n)

            shape_A = torch.from_numpy(shapes[shape_A_idx])
            shape_B = torch.from_numpy(shapes[shape_B_idx])

            geod = approximate_geodesic_distances(shape_B, faces.astype("int"))
            geod /= np.max(geod)

            points_A = area_weighted_normalization(shape_A).to(args.device)
            points_B = area_weighted_normalization(shape_B).to(args.device)

            sep = -torch.ones(points_B.unsqueeze(0).size()[0], 1, 3).to(args.device)
            third_tensor_l = torch.cat((points_A.unsqueeze(0).float(), sep, points_B.unsqueeze(0).float()), 1)
            third_tensor2 = linear1(third_tensor_l)

            dim1 = points_A.unsqueeze(0).shape[1]
            dim2 = points_B.unsqueeze(0).shape[1] + 1

            if args.gaussian_heads:
                shape1_gaussian_attn = gauss_attn(points_A.unsqueeze(0))
                shape2_gaussian_attn = gauss_attn(points_B.unsqueeze(0))
                fixed_attn = torch.zeros((third_tensor2.shape[0], args.gaussian_heads, third_tensor2.shape[1], third_tensor2.shape[1])).to(args.device)
                fixed_attn[:, :, :dim1, :dim1] = shape1_gaussian_attn
                fixed_attn[:, :, dim2:, dim2:] = shape2_gaussian_attn
                if args.force_cross_attn:
                    if args.legacy_force_cross_attn:
                        y_hat_1_m = model(third_tensor2, gaussian_attn=fixed_attn, shape_sep_idx = dim1)
                    else:
                        attn_mask = torch.ones((8, third_tensor2.shape[1], third_tensor2.shape[1])).to(args.device)
                        attn_mask[:args.force_cross_attn, :dim1, :dim1] = 0
                        attn_mask[:args.force_cross_attn, dim2:, dim2:] = 0
                        attn_mask = attn_mask.type(torch.bool)
                        y_hat_1_m = model(third_tensor2, gaussian_attn=fixed_attn, shape_sep_idx=dim1, attn_mask=attn_mask)
                else:
                    y_hat_1_m = model(third_tensor2, gaussian_attn=fixed_attn)
            else:
                y_hat_1_m = model(third_tensor2)

            y_hat1 = linear2(y_hat_1_m)
            y_hat_2 = y_hat1[:, :1000, :]

            y_hat_1 = y_hat1[:, 1001:, :]

            d12 = chamfer_loss(points_A.float(), y_hat_1).to(args.device)
            d21 = chamfer_loss(points_B.float(), y_hat_2).to(args.device)


            if d12 < d21:
                d = torch.cdist(points_A.float(), y_hat_1).squeeze(0).to(args.device)
                ne = get_errors(d, geod)
                err.extend(ne)
            else:
                d = torch.cdist(points_B.float(), y_hat_2).squeeze(0).to(args.device)
                ne = get_errors(d.transpose(1, 0), geod)
                err.extend(ne)

        print("ERROR: ", np.mean(np.array(err)))



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--path_data", default="./dataset/FAUSTS_rem.mat")
    parser.add_argument("--path_model", default="./models")

    parser.add_argument("--run_name", default="custom_trained_model")

    parser.add_argument("--gaussian_heads", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=[], nargs="*")
    
    parser.add_argument("--force_cross_attn", type=int, default=0)
    parser.add_argument("--legacy_force_cross_attn", default=False, action="store_true")

    parser.add_argument("--device", default="auto")

    args, _ = parser.parse_known_args()

    if args.gaussian_heads == 0:
        args.gaussian_heads = False
    elif len(args.sigma) != args.gaussian_heads:
        while len(args.sigma) < args.gaussian_heads:
            if len(args.sigma) > 0:
                args.sigma.append(args.sigma[-1] * 2)
            else:
                args.sigma.append(0.05)
        args.sigma = args.sigma[:args.gaussian_heads]

    if args.force_cross_attn == 0:
        args.force_cross_attn = False

    if args.device == "auto":
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    main(args)
