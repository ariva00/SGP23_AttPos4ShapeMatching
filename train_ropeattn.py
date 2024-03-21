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
from point_gaussian import gauss_attn

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

# ------------------------------------------------------------------------------------------------------------------
# BEGIN SETUP  -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

    set_seed(0)

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
        attn_gaussian_heads=args.gaussian_heads
    ).cuda()

    linear1 = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 64), nn.Tanh(),
                            nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 256), nn.Tanh(), nn.Linear(256, 512)).cuda()

    linear2 = nn.Sequential(nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 64),
                            nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(),
                            nn.Linear(16, 3)).cuda()
    
    params = list(linear1.parameters()) + list(model.parameters()) + list(linear2.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.resume:
        model.load_state_dict(torch.load("models/" + args.run_name))
        linear1.load_state_dict(torch.load("models/l1." + args.run_name))
        linear2.load_state_dict(torch.load("models/l2." + args.run_name))
        optimizer.load_state_dict(torch.load("models/optim." + args.run_name))

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
    lossmse = nn.MSELoss()
    start = time.time()
    for epoch in range(args.n_epoch):
        ep_loss = 0
        for item in tqdm(dataloader_train):
            optimizer.zero_grad(set_to_none=True)

            shapes = item["x"].cuda()
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

            sep = -torch.ones(shape1.shape[0], 1, 3).cuda()

            dim2 = dim1 +1
            inputz = torch.cat((shape1, sep, shape2), 1)

            third_tensor = linear1(inputz)
            if args.gaussian_heads:
                shape1_gaussian_attn = gauss_attn(shape1, args.sigma)
                shape2_gaussian_attn = gauss_attn(shape2, args.sigma)
                fixed_attn = torch.zeros((third_tensor.shape[0], args.gaussian_heads, third_tensor.shape[1], third_tensor.shape[1])).cuda()
                fixed_attn[:, :, :dim1, :dim1] = shape1_gaussian_attn
                fixed_attn[:, :, dim2:, dim2:] = shape2_gaussian_attn
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
        start = time.time()
        print(f"LOSS: {ep_loss} --------------------------------------------------------------------------------------")
        os.makedirs("models", exist_ok=True)

        torch.save(model.state_dict(), "models/" + args.run_name)
        torch.save(linear1.state_dict(), "models/l1." + args.run_name)
        torch.save(linear2.state_dict(), "models/l2." + args.run_name)
        torch.save(optimizer.state_dict(), "models/optim." + args.run_name)


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

    parser.add_argument("--resume", default=False, action="store_true")

    parser.add_argument("--gaussian_heads", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=[0.05], nargs="*")
    parser.add_argument("--no_sep_loss", default=False, action="store_true")

    args = parser.parse_args()

    if args.gaussian_heads == 0:
        args.gaussian_heads = False
    elif len(args.sigma) != args.gaussian_heads:
        while len(args.sigma) < args.gaussian_heads:
            args.sigma.append(args.sigma[-1] * 2)
        args.sigma = args.sigma[:args.gaussian_heads]

    main(args)

