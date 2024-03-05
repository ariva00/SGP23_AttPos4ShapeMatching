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
        rotary_emb_dim=64
    ).cuda()

    linear1 = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 64), nn.Tanh(),
                            nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 256), nn.Tanh(), nn.Linear(256, 512)).cuda()

    linear2 = nn.Sequential(nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 64),
                            nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(),
                            nn.Linear(16, 3)).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            y_hat_l = model(third_tensor)
            y_hat_l2 = linear2(y_hat_l)
            y_hat = y_hat_l2[:, dim2:, :]
            y_hat_b = y_hat_l2[:, :dim1, :]

            loss = ((y_hat[:, gt2, :] - shape1[:, gt1, :]) ** 2).sum() + \
                   ((y_hat_b[:, gt1, :] - shape2[:, gt2, :]) ** 2).sum()+ \
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

    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--n_epoch", default=5000)
    parser.add_argument("--batch_size", default=16)

    parser.add_argument("--path_data", default="dataset/")

    args = parser.parse_args()

    main(args)

