import torch
from tqdm import tqdm
from argparse import ArgumentParser
from utils import  get_errors, chamfer_loss, approximate_geodesic_distances
import numpy as np
import os
import random
import numpy
from model import EncoderPointTransfomer
from datasets import FaustDataset
import torchvision.transforms as transforms
from shape_transforms import NormalizeShape, RandomRotateOneOrAllAxis, NormalizeShapeAreaWeighted, RescaleShape

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    set_seed(0)

    custom_layers = ()

    for i in range(6):
        if i in args.gaussian_blocks:
            custom_layers += ('g', 'f')
        else:
            custom_layers += ('a', 'f')

    dataset = FaustDataset(args.path_data, args.dataset)

    transform = []
    if args.normalize:
        transform.append(NormalizeShape())
    if args.random_rotation:
        transform.append(RandomRotateOneOrAllAxis(360))

    gauss_transform = transform.copy()

    if not args.no_rescale:
        transform.append(RescaleShape(0.741))
    if not args.gauss_no_rescale:
        gauss_transform.append(RescaleShape(0.741))
    transform.append(NormalizeShapeAreaWeighted())
    gauss_transform.append(NormalizeShapeAreaWeighted())

    transform = transforms.Compose(transform)

    n = len(dataset)

    if args.gauss_dataset:
        gauss_dataset = FaustDataset(args.path_data, args.gauss_dataset)

    model = EncoderPointTransfomer(
        heads=args.n_heads,
        dim_head=args.dim_head,
        custom_layers=custom_layers,
        gaussian_heads=args.gaussian_heads,
        inf_gaussian_heads=args.inf_gaussian_heads,
        force_cross_attn=args.force_cross_attn,
        force_self_attn=args.force_self_attn,
        sigma=args.sigma,
        infer_sigma=args.infer_sigma
    ).to(args.device)

    modelname = args.run_name + ".pt"
    pathfolder= args.path_model
    if args.legacy_model:
        model.gauss_attn.load_state_dict(torch.load(os.path.join(pathfolder, "gauss_attn." + modelname), map_location=lambda storage, loc: storage))
        model.linear_in.load_state_dict(torch.load(os.path.join(pathfolder, "l1." + modelname), map_location=lambda storage, loc: storage))
        model.linear_out.load_state_dict(torch.load(os.path.join(pathfolder, "l2." + modelname), map_location=lambda storage, loc: storage))
        model.encoder.load_state_dict(torch.load(os.path.join(pathfolder, modelname), map_location=lambda storage, loc: storage))
    else:
        model.load_state_dict(torch.load(os.path.join(pathfolder, modelname), map_location=lambda storage, loc: storage))
    print(model.gauss_attn.sigmas)

    print(modelname)
    print("MODEL RESUMED ---------------------------------------------------------------------------------------\n")

# ------------------------------------------------------------------------------------------------------------------
# END SETUP  -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

    if args.extended:
        n_pairs = 10000
        pairs = np.meshgrid(range(100), range(100), dtype=int)
        pairs[0] = pairs[0].ravel()
        pairs[1] = pairs[1].ravel()
        pairs = np.array(pairs).T
    else:
        n_pairs = 100
        pairs = np.zeros((n_pairs, 2), dtype=int)
        for i in range(n_pairs):
            shape_A_idx = np.random.randint(n)
            shape_B_idx = np.random.randint(n)
            while shape_A_idx == shape_B_idx:  # avoid taking A exactly equal to B
                shape_B_idx = np.random.randint(n)
            pairs[i, 0] = shape_A_idx
            pairs[i, 1] = shape_B_idx

    model.eval()

    with torch.no_grad():
        err = []
        err_couple = []
        couples = []
        for i in tqdm(range(n_pairs)):

            shape_A_idx = pairs[i, 0]
            shape_B_idx = pairs[i, 1]

            couples.append((shape_A_idx, shape_B_idx))

            shape_A = dataset[shape_A_idx]['x']
            shape_B = dataset[shape_B_idx]['x']
            faces = dataset[shape_A_idx]['faces']

            geod = approximate_geodesic_distances(shape_B.numpy(), faces.numpy())
            geod /= np.max(geod)

            shape_A = shape_A.to(args.device)
            shape_B = shape_B.to(args.device)

            shape_A = transform(shape_A)
            shape_B = transform(shape_B)

            if args.gauss_dataset:
                gauss_shape_A = gauss_dataset[shape_A_idx]['x'].to(args.device)
                gauss_shape_B = gauss_dataset[shape_B_idx]['x'].to(args.device)

                gauss_shape_A = gauss_transform(gauss_shape_A)
                gauss_shape_B = gauss_transform(gauss_shape_B)

            sep = -torch.ones(shape_B.unsqueeze(0).size()[0], 1, 3).to(args.device)

            dim_A = shape_A.unsqueeze(0).shape[1]
            dim_B = shape_A.unsqueeze(0).shape[1]

            if args.random_permutation:
                permidx_A = torch.randperm(dim_A)
                shape_A = shape_A[permidx_A, :]
                gt_A = torch.zeros_like(permidx_A)
                gt_A[permidx_A] = torch.arange(dim_A)

                permidx_B = torch.randperm(dim_B)
                shape_B = shape_B[permidx_B, :]
                gt_B = torch.zeros_like(permidx_B)
                gt_B[permidx_B] = torch.arange(dim_B)

                if args.gauss_dataset:
                    gauss_shape_A = gauss_shape_A[permidx_A, :]
                    gauss_shape_B = gauss_shape_B[permidx_B, :]

            dim_B = dim_A + 1

            x = torch.cat((shape_A.unsqueeze(0).float(), sep, shape_B.unsqueeze(0).float()), 1)

            y = model(x, mask_head=args.mask_head)
            y_shape_A = y[:, dim_B:, :] # shape_B points in shape_A space
            y_shape_B = y[:, :dim_A, :] # shape_A points in shape_B space

            if args.random_permutation:
                y_shape_A = y_shape_A[:, gt_B, :]
                y_shape_B = y_shape_B[:, gt_A, :]
                shape_B = shape_B[gt_B, :]
                shape_A = shape_A[gt_A, :]

            d_A = chamfer_loss(shape_A.float(), y_shape_A).to(args.device)
            d_B = chamfer_loss(shape_B.float(), y_shape_B).to(args.device)

            if d_A < d_B:
                d = torch.cdist(shape_A.float(), y_shape_A).squeeze(0).to(args.device)
                ne = get_errors(d, geod)
                err_couple.append(np.sum(ne))
                err.append(ne)
            else:
                d = torch.cdist(shape_B.float(), y_shape_B).squeeze(0).to(args.device)
                ne = get_errors(d.transpose(1, 0), geod)
                err_couple.append(np.sum(ne))
                err.append(ne)

        print("ERROR MIN: ", np.array(err).min())
        print("ERROR MAX: ", np.array(err).max())
        print("ERROR MEAN: ", np.mean(np.array(err)))
        print("ERROR VAR: ", np.var(np.array(err)))

        return np.array(err), np.array(err_couple), np.array(couples)



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--run_name", default="custom_trained_model", help="name of the run, determines the name of the saved model")

    parser.add_argument("--path_data", default="dataset/", help="path to dir containing the dataset")
    parser.add_argument("--path_model", default="./models", help="path to dir where the model will be saved")
    parser.add_argument("--dataset", default="FAUSTS_rem", help="name of the dataset")

    parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads (Including Gaussian Heads)")
    parser.add_argument("--dim_head", type=int, default=64, help="dimension of the attention heads")

    parser.add_argument("--gaussian_heads", type=int, default=0, help="number of gaussian attention heads")
    parser.add_argument("--sigma", type=float, default=[], nargs="*", help="initial sigma for the gaussian attention heads")
    parser.add_argument("--infer_sigma", default=False, action="store_true", help="layer output to use to infer the sigma of the subsequent gaussian attention heads")

    parser.add_argument("--force_cross_attn", type=int, default=0, help="masks the self attention part of the dot-product attention heads")
    parser.add_argument("--force_self_attn", type=int, default=0, help="masks the self attention part of the dot-product attention heads")

    parser.add_argument("--inf_gaussian_heads", type=int, default=0, help="number of infinite gaussian attention heads, these heads have a uniform attention for all points")

    parser.add_argument("--condition_self", type=int, default=0, help="number of heads to condition to self attention")
    parser.add_argument("--condition_cross", type=int, default=0, help="number of heads to condition to cross attention")
    parser.add_argument("--condition_mask", default=False, action="store_true", help="mask the conditioned heads to only condition the correct diagonals of the attention matrices")

    parser.add_argument("--device", default="auto", help="device to use for training, auto will use cuda if available, mps if available, else cpu")

    parser.add_argument("--mask_head", type=int, default=[], nargs="*", help="masks the attention heads at the specified indices")

    parser.add_argument("--no_rescale", default=False, action="store_true", help="do not rescale the shapes")
    parser.add_argument("--gauss_dataset", default=None, help="name of the dataset to use for the gaussian attention euclidean distances (if different from the main dataset)")
    parser.add_argument("--gauss_no_rescale", default=False, action="store_true", help="do not rescale the shapes of the gauss_dataset")

    parser.add_argument("--normalize", default=False, action="store_true", help="normalize the input shapes to the range [-1, 1]")

    parser.add_argument("--random_rotation", default=False, action="store_true", help="apply random rotation to the shapes")
    parser.add_argument("--random_permutation", default=False, action="store_true", help="apply random permutation to the shapes points")

    parser.add_argument("--gaussian_blocks", type=int, default=list(range(6)), nargs="*", help="blocks to use gaussian attention in, the default is in all blocks")

    parser.add_argument("--extended", default=False, action="store_true", help="use an extended set of pairs for evaluation")

    parser.add_argument("--legacy_model", default=False, action="store_true", help="use the legacy model save format (four different files instead of one)")

    args, _ = parser.parse_known_args()

    if args.gaussian_heads == 0:
        args.gaussian_heads = False
    elif args.infer_sigma:
        args.sigma = []
    elif len(args.sigma) != args.gaussian_heads:
        while len(args.sigma) < args.gaussian_heads:
            if len(args.sigma) > 0:
                args.sigma.append(args.sigma[-1] * 2)
            else:
                args.sigma.append(0.05)
        args.sigma = args.sigma[:args.gaussian_heads]

    if args.force_cross_attn == 0:
        args.force_cross_attn = False

    if args.condition_self == 0:
        args.condition_self = False
    if args.condition_cross == 0:
        args.condition_cross = False

    if args.condition_mask:
        args.force_cross_attn = args.condition_cross
        args.force_self_attn = args.condition_self

    if args.device == "auto":
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    main(args)
