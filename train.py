import os
import time
from torch.utils.data import DataLoader
from datasets import SMPLDataset
from utils import approximate_geodesic_distances, get_errors, chamfer_loss
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import torch.nn as nn
import numpy as np
import random
import numpy
import logging
import torchvision.transforms as transforms
from shape_transforms import RandomRotateOneOrAllAxis, NormalizeShapeAreaWeighted, NormalizeShape, GaussianNoise
from model import EncoderPointTransfomer
from point_gaussian import gauss_attn, estimate_sigmas, gauss_loss

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    logger = logging.getLogger(args.run_name)
    logger.info(f"training {args.run_name}")
    logger.info(f"args: {args}")
    logger.info(f"initial sigma: {args.sigma}")

# ------------------------------------------------------------------------------------------------------------------
# BEGIN SETUP  -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

    set_seed(0)

    custom_layers = ()

    for i in range(6):
        if i in args.gaussian_blocks:
            custom_layers += ('g', 'f')
        else:
            custom_layers += ('a', 'f')

    # TRANSFORM
    transform_train = []
    transform_train.append(RandomRotateOneOrAllAxis(360))
    transform_train.append(NormalizeShapeAreaWeighted())
    if args.normalize:
        transform_train.append(NormalizeShape())
    if args.noise:
        if args.noise_p == 1:
            transform_train.append(GaussianNoise(args.noise))
        else:
            transform_train.append(transforms.RandomApply([GaussianNoise(args.noise)], p=args.noise_p))
    transform_train = transforms.Compose(transform_train)

    transform_test = []
    transform_test.append(NormalizeShapeAreaWeighted())
    if args.normalize:
        transform_test.append(NormalizeShape())
    transform_test = transforms.Compose(transform_test)

    # DATASET
    data_train = SMPLDataset(args.path_data, train=True, transform=transform_train)
    data_test = SMPLDataset(args.path_data, train=False, transform=transform_test)

    # DATALOADERS
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, drop_last=True)
    num_points = 1000

    # INITIALIZE MODEL
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

    if args.learn_sigma:
        params = [
            { "params": list(model.linear_in.parameters()) + list(model.encoder.parameters()) + list(model.linear_out.parameters()) },
            { "params": model.gauss_attn.parameters(), "lr": args.lr * args.lr_mult}
        ]
    else:
        for p in model.gauss_attn.parameters():
            p.requires_grad = False
        params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.path_model, args.run_name + ".pt"), map_location=lambda storage, loc: storage))
        optimizer.load_state_dict(torch.load(os.path.join(args.path_model, "optim." + args.run_name + ".pt"), map_location=lambda storage, loc: storage))

    initial_sigma = model.gauss_attn.sigmas.clone().detach().cpu()
    print("initial sigma: ", initial_sigma)

# ------------------------------------------------------------------------------------------------------------------
# END SETUP  -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
# BEGIN TRAINING ---------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

    print("TRAINING --------------------------------------------------------------------------------------------------")
    model = model.train()

    best_loss = float("inf")

    for epoch in range(args.n_epoch):
        logger.info(f"starting epoch {epoch}/{args.n_epoch-1}")
        start = time.time()
        epoch_loss = train(model, dataloader_train, optimizer, num_points, args)
        print(f"EPOCH: {epoch} HAS FINISHED, in {time.time() - start} SECONDS! ---------------------------------------")
        print(f"LOSS: {epoch_loss} --------------------------------------------------------------------------------------")
        os.makedirs(args.path_model, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(args.path_model, args.run_name + ".pt"))
        torch.save(optimizer.state_dict(), os.path.join(args.path_model, "optim." + args.run_name + ".pt"))

        # VALIDATION

        if args.use_validation:
            if ((epoch + 1) % args.validation_step == 0 or epoch == args.n_epoch - 1):
                model = model.eval()
                err = test(model, dataloader_test, args)
                model = model.train()
                logger.info(f"ending epoch {epoch}/{args.n_epoch-1}, time {time.time() - start} seconds, loss {epoch_loss}, val err {err}")
                print(f"VALIDATION ERR: {err} ---------------------------------------------------------------------------------")
                
                if args.save_best and err < best_loss:
                    best_loss = err
                    torch.save(model.state_dict(), os.path.join(args.path_model, "best." + args.run_name + ".pt"))
                    torch.save(optimizer.state_dict(), os.path.join(args.path_model, "optim." + "best." + args.run_name + ".pt"))
                    logger.info(f"new best epoch {epoch}/{args.n_epoch-1}, val err {err}")
            else:
                logger.info(f"ending epoch {epoch}/{args.n_epoch-1}, time {time.time() - start} seconds, loss {epoch_loss}")
        else:
            logger.info(f"ending epoch {epoch}/{args.n_epoch-1}, time {time.time() - start} seconds, loss {epoch_loss}")

            if args.save_best and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(args.path_model, "best." + args.run_name + ".pt"))
                torch.save(optimizer.state_dict(), os.path.join(args.path_model, "optim." + "best." + args.run_name + ".pt"))
                logger.info(f"new best epoch {epoch}/{args.n_epoch-1}, loss {epoch_loss}")

    logger.info(f"initial sigma: {initial_sigma}")
    logger.info(f"final sigma: {model.gauss_attn.sigmas.clone().detach().cpu()}")
    logger.info(f"training {args.run_name} has finished")

    print("initial sigma: ", initial_sigma)
    print("final sigma: ", model.gauss_attn.sigmas.clone().detach().cpu())


# ------------------------------------------------------------------------------------------------------------------
# END TRAINING -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

def train(model, dataloader, optimizer, num_points, args):

    epoch_loss = 0
    geod_dist = None
    for item in tqdm(dataloader):
        if geod_dist is None:
            geod_dist = torch.tensor(approximate_geodesic_distances(item['y'][0].cpu().numpy(), item['faces'][0].cpu().numpy())).to(args.device)
        optimizer.zero_grad(set_to_none=True)

        # DATA PREPARATION

        shapes = item["x"].to(args.device)
        shape_A = shapes[:args.batch_size // 2, :, :]
        shape_B = shapes[args.batch_size // 2:, :, :]

        dim_A = num_points
        permidx_A = torch.randperm(dim_A)
        shape_A = shape_A[:, permidx_A, :]
        gt_A = torch.zeros_like(permidx_A)
        gt_A[permidx_A] = torch.arange(dim_A)

        dim_B = num_points
        permidx_B = torch.randperm(dim_B)
        shape_B = shape_B[:, permidx_B, :]
        gt_B = torch.zeros_like(permidx_B)
        gt_B[permidx_B] = torch.arange(dim_B)

        sep = -torch.ones(shape_A.shape[0], 1, 3).to(args.device)

        dim_B = dim_A +1
        x = torch.cat((shape_A, sep, shape_B), 1)

        # FORWARD

        if args.condition_self or args.condition_cross:
            y, hiddens = model(x, return_hiddens=True)
        else:
            y = model(x)
        y_shape_A = y[:, dim_B:, :] # shape_B points in shape_A space
        y_shape_B = y[:, :dim_A, :] # shape_A points in shape_B space

        # LOSS CONDITIONING

        if args.condition_self or args.condition_cross:
            post_softmax_attn = hiddens.attn_intermediates[args.condition_layer].post_softmax_attn

            if args.condition_self and args.condition_loss in ("diff", "cos"):
                if args.condition_fixed:
                    sigmas_AA = torch.tensor(args.sigma[:args.condition_self]).to(args.device)
                    sigmas_BB = torch.tensor(args.sigma[:args.condition_self]).to(args.device)
                else:
                    if args.geod_dist:
                        sigmas_AA = estimate_sigmas(shape_A, post_softmax_attn[:, -args.condition_self:, :dim_A, :dim_A], geod_dist[permidx_A, :][:, permidx_A])
                        sigmas_BB = estimate_sigmas(shape_B, post_softmax_attn[:, -args.condition_self:, dim_B:, dim_B:], geod_dist[permidx_B, :][:, permidx_B])
                    else:
                        sigmas_AA = estimate_sigmas(shape_A, post_softmax_attn[:, -args.condition_self:, :dim_A, :dim_A])
                        sigmas_BB = estimate_sigmas(shape_B, post_softmax_attn[:, -args.condition_self:, dim_B:, dim_B:])

                if args.geod_dist:
                    attn_AA = gauss_attn(shape_A, sigmas_AA.detach(), geod_dist[permidx_A, :][:, permidx_A])
                    attn_BB = gauss_attn(shape_B, sigmas_BB.detach(), geod_dist[permidx_B, :][:, permidx_B])
                else:
                    attn_AA = gauss_attn(shape_A, sigmas_AA.detach())
                    attn_BB = gauss_attn(shape_B, sigmas_BB.detach())
            
            if args.condition_cross and args.condition_loss in ("diff", "cos"):
                if args.condition_fixed:
                    sigmas_AB = torch.tensor(args.sigma[args.condition_self:]).to(args.device)
                    sigmas_BA = torch.tensor(args.sigma[args.condition_self:]).to(args.device)
                else:
                    if args.geod_dist:
                        sigmas_AB = estimate_sigmas((shape_A[:, gt_A, :])[:, permidx_B, :], post_softmax_attn[:, :args.condition_cross, dim_B:, :dim_A], geod_dist[permidx_B, :][:, permidx_B])
                        sigmas_BA = estimate_sigmas((shape_B[:, gt_B, :])[:, permidx_A, :], post_softmax_attn[:, :args.condition_cross, :dim_A, dim_B:], geod_dist[permidx_A, :][:, permidx_A])
                    else:
                        sigmas_AB = estimate_sigmas((shape_A[:, gt_A, :])[:, permidx_B, :], post_softmax_attn[:, :args.condition_cross, dim_B:, :dim_A])
                        sigmas_BA = estimate_sigmas((shape_B[:, gt_B, :])[:, permidx_A, :], post_softmax_attn[:, :args.condition_cross, :dim_A, dim_B:])

                if args.geod_dist:
                    attn_AB = gauss_attn((shape_A[:, gt_A, :])[:, permidx_B, :], sigmas_AB.detach(), geod_dist[permidx_B, :][:, permidx_B])[:, :, :, gt_B][:, :, :, permidx_A]
                    attn_BA = gauss_attn((shape_B[:, gt_B, :])[:, permidx_A, :], sigmas_BA.detach(), geod_dist[permidx_A, :][:, permidx_A])[:, :, :, gt_A][:, :, :, permidx_B]
                else:
                    attn_AB = gauss_attn((shape_A[:, gt_A, :])[:, permidx_B, :], sigmas_AB.detach())[:, :, :, gt_B][:, :, :, permidx_A]
                    attn_BA = gauss_attn((shape_B[:, gt_B, :])[:, permidx_A, :], sigmas_BA.detach())[:, :, :, gt_A][:, :, :, permidx_B]
            if args.condition_loss == "diff":
                attn_loss = torch.empty(0, device=args.device)
                if args.condition_self:
                    attn_loss = torch.cat((
                        attn_loss,
                        (post_softmax_attn[:, -args.condition_self:, :dim_A, :dim_A] - attn_AA.softmax(dim=-1)).abs().sum().reshape(1),
                        (post_softmax_attn[:, -args.condition_self:, dim_B:, dim_B:] - attn_BB.softmax(dim=-1)).abs().sum().reshape(1)
                    ))
                if args.condition_cross:
                    attn_loss = torch.cat((
                        attn_loss,
                        (post_softmax_attn[:, :args.condition_cross, dim_B:, :dim_A] - attn_AB.softmax(dim=-1)).abs().sum().reshape(1),
                        (post_softmax_attn[:, :args.condition_cross, :dim_A, dim_B:] - attn_BA.softmax(dim=-1)).abs().sum().reshape(1)
                    ))
                attn_loss = attn_loss.nanmean()

            elif args.condition_loss == "cos":
                attn_loss = torch.empty(0, device=args.device)
                if args.condition_self:
                    attn_loss = torch.cat((
                        attn_loss,
                        (post_softmax_attn.shape[0] * args.condition_self * post_softmax_attn.shape[2]) - nn.functional.cosine_similarity(post_softmax_attn[:, -args.condition_self:, :dim_A, :dim_A], attn_AA.softmax(dim=-1), dim = 2).sum().reshape(1),
                        (post_softmax_attn.shape[0] * args.condition_self * post_softmax_attn.shape[2]) - nn.functional.cosine_similarity(post_softmax_attn[:, -args.condition_self:, dim_B:, dim_B:], attn_BB.softmax(dim=-1), dim = 2).sum().reshape(1),
                    ))
                if args.condition_cross:
                    attn_loss = torch.cat((
                        attn_loss,
                        (post_softmax_attn.shape[0] * args.condition_cross * post_softmax_attn.shape[2]) - nn.functional.cosine_similarity(post_softmax_attn[:, :args.condition_cross, dim_B:, :dim_A], attn_AB.softmax(dim=-1), dim = 2).sum().reshape(1),
                        (post_softmax_attn.shape[0] * args.condition_cross * post_softmax_attn.shape[2]) - nn.functional.cosine_similarity(post_softmax_attn[:, :args.condition_cross, :dim_A, dim_B:], attn_BA.softmax(dim=-1), dim = 2).sum().reshape(1),
                    ))
                attn_loss = attn_loss.mean()

            elif args.condition_loss == "sort":
                attn_loss = 0
                if args.condition_self:
                    attn_loss += gauss_loss(shape_A, post_softmax_attn[:, -args.condition_self:, :dim_A, :dim_A]).sum()
                    attn_loss += gauss_loss(shape_B, post_softmax_attn[:, -args.condition_self:, dim_B:, dim_B:]).sum()
                if args.condition_cross:
                    attn_loss += gauss_loss((shape_A[:, gt_A, :])[:, permidx_B, :], post_softmax_attn[:, :args.condition_cross, dim_B:, :dim_A][:,:,:, gt_A][:,:,:, permidx_B]).sum()
                    attn_loss += gauss_loss((shape_B[:, gt_B, :])[:, permidx_A, :], post_softmax_attn[:, :args.condition_cross, :dim_A, dim_B:][:,:,:, gt_B][:,:,:, permidx_A]).sum()

        # LOSS COMPUTATION

        if args.no_sep_loss:
            loss = ((y_shape_A[:, gt_B, :] - shape_A[:, gt_A, :]) ** 2).sum() + \
                    ((y_shape_B[:, gt_A, :] - shape_B[:, gt_B, :]) ** 2).sum()
        else:
            loss = ((y_shape_A[:, gt_B, :] - shape_A[:, gt_A, :]) ** 2).sum() + \
                    ((y_shape_B[:, gt_A, :] - shape_B[:, gt_B, :]) ** 2).sum() + \
                    nn.functional.mse_loss(y[:, dim_A, :], sep[:, 0, :])

        if args.condition_self or args.condition_cross:
            if args.condition_self and args.condition_fds:
                attn_loss += cross_heads_loss(post_softmax_attn[:, -args.condition_self:, :dim_A, :dim_A])
                attn_loss += cross_heads_loss(post_softmax_attn[:, -args.condition_self:, dim_B:, dim_B:])
            
            if args.condition_cross and args.condition_fds:
                attn_loss += cross_heads_loss(post_softmax_attn[:, :args.condition_cross, dim_B:, :dim_A])
                attn_loss += cross_heads_loss(post_softmax_attn[:, :args.condition_cross, :dim_A, dim_B:])

            loss += attn_loss

        if torch.isnan(loss):
            print("\nNAN LOSS\n")
            exit()

        # BACKPROPAGATION

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss

def test(model, dataloader, args):
    err = []
    for item in tqdm(dataloader):
        shapes = item[0].to(args.device)
        faces = item[1]
        shape_A = shapes[:shapes.shape[0] // 2, :, :]
        shape_B = shapes[shapes.shape[0] // 2:, :, :]

        dim_A = shape_A.shape[1]
        dim_B = shape_B.shape[1]

        sep = -torch.ones(shape_A.shape[0], 1, 3).to(args.device)

        dim_B = dim_A +1
        x = torch.cat((shape_A, sep, shape_B), 1)
        y = model(x)

        y_shape_A = y[:, dim_B:, :] # shape_B points in shape_A space
        y_shape_B = y[:, :dim_A, :] # shape_A points in shape_B space


        d_A = chamfer_loss(shape_A, y_shape_A).cpu()
        d_B = chamfer_loss(shape_B, y_shape_B).cpu()

        dist_A = torch.cdist(shape_A.float(), y_shape_A).cpu()
        dist_B = torch.cdist(shape_B.float(), y_shape_B).cpu()

        for i in range(shape_A.shape[0]):
            geod = approximate_geodesic_distances(shape_B[i].cpu(), faces[i].numpy())
            geod /= np.max(geod)

            if d_A[i] < d_B[i]:
                ne = get_errors(dist_A[i], geod)
                err.append(ne)
            else:
                ne = get_errors(dist_B[i].transpose(1, 0), geod)
                err.append(ne)
    return np.mean(np.array(err))

def cross_heads_loss(attn:torch.Tensor):
    return (-(attn - attn.roll(1, 1)).abs()).exp().mean(dim=2).sum()

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--run_name", default="custom_trained_model", help="name of the run, determines the name of the saved model")

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=5000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")

    parser.add_argument("--path_data", default="dataset/", help="path to dir containing the dataset")
    parser.add_argument("--path_model", default="./models", help="path to dir where the model will be saved")

    parser.add_argument("--resume", default=False, action="store_true", help="resume training from a saved model, the model is determined by run_name")

    parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads (Including Gaussian Heads)")
    parser.add_argument("--dim_head", type=int, default=64, help="dimension of the attention heads")

    parser.add_argument("--gaussian_heads", type=int, default=0, help="number of gaussian attention heads")
    parser.add_argument("--sigma", type=float, default=[], nargs="*", help="initial sigma for the gaussian attention heads")
    parser.add_argument("--no_sep_loss", default=False, action="store_true", help="do not use additional loss term on the separator")
    parser.add_argument("--learn_sigma", default=False, action="store_true", help="learn the sigma of the gaussian attention heads")
    parser.add_argument("--lr_mult", type=float, default=1.0, help="learning rate multiplier for the sigma parameters")
    parser.add_argument("--infer_sigma", default=False, action="store_true", help="layer output to use to infer the sigma of the subsequent gaussian attention heads")

    parser.add_argument("--force_cross_attn", type=int, default=0, help="masks the self attention part of the dot-product attention heads")
    parser.add_argument("--force_self_attn", type=int, default=0, help="masks the self attention part of the dot-product attention heads")

    parser.add_argument("--inf_gaussian_heads", type=int, default=0, help="number of infinite gaussian attention heads, these heads have a uniform attention for all points")

    parser.add_argument("--condition_self", type=int, default=0, help="number of heads to condition to self attention")
    parser.add_argument("--condition_cross", type=int, default=0, help="number of heads to condition to cross attention")
    parser.add_argument("--condition_loss", default="diff", help="loss to use for conditioning, one of 'diff' (for difference), 'cos' (for cosine similarity), 'sort' (for sorting)")
    parser.add_argument("--condition_layer", type=int, default=5, help="layer to condition the attention weights")
    parser.add_argument("--condition_fixed", default=False, action="store_true", help="use fixed sigmas for conditioning, the conditioning is not learned. If True, the sigmas are the ones in the sigma argument from the self ones to the cross ones in order, if False, the sigmas are estimated from the attention weights")
    parser.add_argument("--condition_mask", default=False, action="store_true", help="mask the conditioned heads to only condition the correct diagonals of the attention matrices")
    parser.add_argument("--condition_fds", default=False, action="store_true", help="condition the heads to produce different sigmas")

    parser.add_argument("--device", default="auto", help="device to use for training, auto will use cuda if available, mps if available, else cpu")


    parser.add_argument("--log_file", default="train.log", help="file to log the training process")

    parser.add_argument("--gaussian_blocks", type=int, default=list(range(6)), nargs="*", help="blocks to use gaussian attention in, the default is in all blocks")

    parser.add_argument("--save_best", default=False, action="store_true", help="save the model with the best training loss")

    parser.add_argument("--normalize", default=False, action="store_true", help="normalize the input shapes to the range [-1, 1]")
    parser.add_argument("--noise", type=float, default=0.0, help="add noise to the input shapes")
    parser.add_argument("--noise_p", type=float, default=0.5, help="probability of adding noise to a shape")

    parser.add_argument("--geod_dist", default=False, action="store_true", help="use geodesic distances to condition via loss")

    parser.add_argument("--use_validation", default=False, action="store_true", help="use a validation set to evaluate the model")
    parser.add_argument("--validation_step", type=int, default=5, help="number of epochs between validation evaluations")


    args, _ = parser.parse_known_args()

    if args.gaussian_heads == 0:
        args.gaussian_heads = False
    elif args.infer_sigma:
        args.sigma = []
    elif len(args.sigma) != args.gaussian_heads:
        while len(args.sigma) < args.gaussian_heads:
            if args.learn_sigma:
                args.sigma.append(torch.rand(1).item())
            elif len(args.sigma) > 0:
                args.sigma.append(args.sigma[-1] * 2)
            else:
                args.sigma.append(0.05)
        args.sigma = args.sigma[:args.gaussian_heads]
    if args.force_cross_attn == 0:
        args.force_cross_attn = False
    if args.condition_fixed:
        assert len(args.sigma) == args.condition_self + args.condition_cross, "The number of sigmas must match the number of conditioned heads"

    if args.condition_self == 0:
        args.condition_self = False
    if args.condition_cross == 0:
        args.condition_cross = False

    if args.condition_mask:
        args.force_cross_attn = args.condition_cross
        args.force_self_attn = args.condition_self

    if args.noise == 0:
        args.noise = False

    if args.device == "auto":
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    main(args)
