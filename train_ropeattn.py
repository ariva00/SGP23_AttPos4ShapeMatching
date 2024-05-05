import os
import time
from torch.utils.data import DataLoader
from transmatching.Data.dataset_smpl import SMPLDataset
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from x_transformers import Encoder
import torch.nn as nn
import random
import numpy
from point_gaussian import GaussianAttention
import logging

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

    # DATASET
    data_train = SMPLDataset(args.path_data, train=True)

    # DATALOADERS
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    num_points = 1000

    # INITIALIZE MODEL
    model = Encoder(
        dim=512,
        depth=6,
        heads=8,
        dim_head_custom = 64,
        pre_norm=False,
        residual_attn=True,
        rotary_pos_emb=True,
        rotary_emb_dim=64,
        custom_layers=custom_layers,
        gauss_gaussian_heads=args.gaussian_heads + args.inf_gaussian_heads,
        attn_force_cross_attn=args.force_cross_attn,
        attn_legacy_force_cross_attn=args.legacy_force_cross_attn,
    ).to(args.device)

    gauss_attn = GaussianAttention(args.sigma).to(args.device)

    linear1 = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 64), nn.Tanh(),
                            nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 256), nn.Tanh(), nn.Linear(256, 512)).to(args.device)

    linear2 = nn.Sequential(nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 64),
                            nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(),
                            nn.Linear(16, 3)).to(args.device)

    if args.learn_sigma:
        params = [
            { "params": list(linear1.parameters()) + list(model.parameters()) + list(linear2.parameters()) },
            { "params": gauss_attn.parameters(), "lr": args.lr * args.lr_mult}
        ]
    else:
        for p in gauss_attn.parameters():
            p.requires_grad = False
        params = list(linear1.parameters()) + list(model.parameters()) + list(linear2.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.path_model, args.run_name + ".pt")))
        linear1.load_state_dict(torch.load(os.path.join(args.path_model, "l1." + args.run_name + ".pt")))
        linear2.load_state_dict(torch.load(os.path.join(args.path_model, "l2." + args.run_name + ".pt")))
        optimizer.load_state_dict(torch.load(os.path.join(args.path_model, "optim." + args.run_name + ".pt")))
        gauss_attn.load_state_dict(torch.load(os.path.join(args.path_model, "gauss_attn." + args.run_name + ".pt")))

    initial_sigma = gauss_attn.sigmas.clone().detach().cpu()
    print("initial sigma: ", initial_sigma)

# ------------------------------------------------------------------------------------------------------------------
# END SETUP  -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
# BEGIN TRAINING ---------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

    print("TRAINING --------------------------------------------------------------------------------------------------")
    model = model.train()
    linear1 = linear1.train()
    linear2 = linear2.train()
    gauss_attn = gauss_attn.train()
    lossmse = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(args.n_epoch):
        logger.info(f"starting epoch {epoch}/{args.n_epoch-1}")
        start = time.time()
        ep_loss = 0
        for item in tqdm(dataloader_train):
            optimizer.zero_grad(set_to_none=True)

            shapes = item["x"].to(args.device)
            shape1 = shapes[:args.batch_size // 2, :, :]
            shape2 = shapes[args.batch_size // 2:, :, :]

            dim1 = num_points
            permidx1 = torch.randperm(dim1)
            shape1 = shape1[:, permidx1, :]
            gt1 = torch.zeros_like(permidx1)
            gt1[permidx1] = torch.arange(dim1)

            dim2 = num_points
            permidx2 = torch.randperm(dim2)
            shape2 = shape2[:, permidx2, :]
            gt2 = torch.zeros_like(permidx2)
            gt2[permidx2] = torch.arange(dim2)

            sep = -torch.ones(shape1.shape[0], 1, 3).to(args.device)

            dim2 = dim1 +1
            inputz = torch.cat((shape1, sep, shape2), 1)

            third_tensor = linear1(inputz)
            if args.gaussian_heads or args.inf_gaussian_heads:
                fixed_attn = torch.zeros((third_tensor.shape[0], args.gaussian_heads + args.inf_gaussian_heads, third_tensor.shape[1], third_tensor.shape[1])).to(args.device)
                if args.gaussian_heads:
                    shape1_gaussian_attn = gauss_attn(shape1)
                    shape2_gaussian_attn = gauss_attn(shape2)
                    fixed_attn[:, args.inf_gaussian_heads:, :dim1, :dim1] = shape1_gaussian_attn
                    fixed_attn[:, args.inf_gaussian_heads:, dim2:, dim2:] = shape2_gaussian_attn
                if args.inf_gaussian_heads:
                    fixed_attn[:, :args.inf_gaussian_heads, :dim1, :dim1] = 1
                    fixed_attn[:, :args.inf_gaussian_heads, dim2:, dim2:] = 1
                if args.force_cross_attn:
                    if args.legacy_force_cross_attn:
                        y_hat_l = model(third_tensor, gaussian_attn=fixed_attn, shape_sep_idx = dim1)
                    else:
                        attn_mask = torch.ones((8, third_tensor.shape[1], third_tensor.shape[1])).to(args.device)
                        attn_mask[:args.force_cross_attn, :dim1, :dim1] = 0
                        attn_mask[:args.force_cross_attn, dim2:, dim2:] = 0
                        attn_mask = attn_mask.type(torch.bool)
                        y_hat_l = model(third_tensor, gaussian_attn=fixed_attn, shape_sep_idx=dim1, attn_mask=attn_mask)
                else:
                    y_hat_l = model(third_tensor, gaussian_attn=fixed_attn)
            else:
                y_hat_l = model(third_tensor)
            y_hat_l2 = linear2(y_hat_l)
            y_hat = y_hat_l2[:, dim2:, :]
            y_hat_b = y_hat_l2[:, :dim1, :]

            if args.no_sep_loss:
                loss = ((y_hat[:, gt2, :] - shape1[:, gt1, :]) ** 2).sum() + \
                       ((y_hat_b[:, gt1, :] - shape2[:, gt2, :]) ** 2).sum()
            else:
                loss = ((y_hat[:, gt2, :] - shape1[:, gt1, :]) ** 2).sum() + \
                       ((y_hat_b[:, gt1, :] - shape2[:, gt2, :]) ** 2).sum() + \
                       lossmse(y_hat_l2[:, dim1, :],sep[:, 0, :])

            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

        print(f"EPOCH: {epoch} HAS FINISHED, in {time.time() - start} SECONDS! ---------------------------------------")
        print(f"LOSS: {ep_loss} --------------------------------------------------------------------------------------")
        os.makedirs(args.path_model, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(args.path_model, args.run_name + ".pt"))
        torch.save(linear1.state_dict(), os.path.join(args.path_model, "l1." + args.run_name + ".pt"))
        torch.save(linear2.state_dict(), os.path.join(args.path_model, "l2." + args.run_name + ".pt"))
        torch.save(optimizer.state_dict(), os.path.join(args.path_model, "optim." + args.run_name + ".pt"))
        torch.save(gauss_attn.state_dict(), os.path.join(args.path_model, "gauss_attn." + args.run_name + ".pt"))

        logger.info(f"ending epoch {epoch}/{args.n_epoch-1}, time {time.time() - start} seconds, loss {ep_loss}")

        if args.save_best and ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(model.state_dict(), os.path.join(args.path_model, "best." + args.run_name + ".pt"))
            torch.save(linear1.state_dict(), os.path.join(args.path_model, "l1." + "best." + args.run_name + ".pt"))
            torch.save(linear2.state_dict(), os.path.join(args.path_model, "l2." + "best." + args.run_name + ".pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.path_model, "optim." + "best." + args.run_name + ".pt"))
            torch.save(gauss_attn.state_dict(), os.path.join(args.path_model, "gauss_attn." + "best." + args.run_name + ".pt"))
            logger.info(f"new best epoch {epoch}/{args.n_epoch-1}, loss {ep_loss}")

    logger.info(f"initial sigma: {initial_sigma}")
    logger.info(f"final sigma: {gauss_attn.sigmas.clone().detach().cpu()}")
    logger.info(f"training {args.run_name} has finished")

    print("initial sigma: ", initial_sigma)
    print("final sigma: ", gauss_attn.sigmas.clone().detach().cpu())


# ------------------------------------------------------------------------------------------------------------------
# END TRAINING -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--run_name", default="custom_trained_model")

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epoch", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--path_data", default="dataset/")
    parser.add_argument("--path_model", default="./models")

    parser.add_argument("--resume", default=False, action="store_true")

    parser.add_argument("--gaussian_heads", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=[], nargs="*")
    parser.add_argument("--no_sep_loss", default=False, action="store_true")
    parser.add_argument("--learn_sigma", default=False, action="store_true")

    parser.add_argument("--force_cross_attn", type=int, default=0)
    parser.add_argument("--legacy_force_cross_attn", default=False, action="store_true")

    parser.add_argument("--inf_gaussian_heads", type=int, default=0)

    parser.add_argument("--device", default="auto")

    parser.add_argument("--lr_mult", type=float, default=1.0)

    parser.add_argument("--log_file", default="train.log")

    parser.add_argument("--gaussian_blocks", type=int, default=list(range(6)), nargs="*")

    parser.add_argument("--save_best", default=False, action="store_true")

    args, _ = parser.parse_known_args()

    if args.gaussian_heads == 0:
        args.gaussian_heads = False
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

    if args.device == "auto":
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    main(args)
