import numpy as np
import os
import torch
import trimesh
from torch.utils.data import Dataset
from scipy.io import loadmat
from transmatching.Utils.utils import est_area

# Original transmatching repository https://github.com/GiovanniTRA/transmatching
# The orignial code is distributed under the MIT license reported in the license folder

class SMPLDataset(Dataset):

    def __init__(self, in_path, train=True, transform=None):
        self.in_path = in_path
        self.train = train
        self.train_data = torch.from_numpy(np.load(os.path.join(self.in_path, '12k_shapes_train.npy'))).float()
        self.test_data = torch.from_numpy(np.load(os.path.join(self.in_path, '12k_shapes_test.npy'))).float()
        self.faces = torch.from_numpy(trimesh.load_mesh((os.path.join(self.in_path, '12ktemplate.ply')), process=False).faces).int()
        self.ref = torch.from_numpy(trimesh.load_mesh((os.path.join(self.in_path, '12ktemplate.ply')), process=False).vertices).float()
        self.transform = transform

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        return self.test_data.shape[0]

    def __getitem__(self, index):
        if self.train:
            shape = self.train_data[index]
        else:
            shape = self.test_data[index]

        if self.transform:
            shape = self.transform(shape)

        return {'x':shape, 'faces':self.faces, 'y':self.ref}

class FaustDataset(Dataset):

    def __init__(self, in_path, dataset="FAUSTS_rem", area=True, transform=None):
        self.in_path = in_path
        self.area = area
        self.mat = loadmat(self.in_path + dataset + ".mat")
        self.data = torch.from_numpy(self.mat["vertices"]).float()
        self.faces = torch.from_numpy(self.mat["f"]).int() - 1
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        shape = self.data[index]
        shape = shape * 0.7535

        if self.area:
            A = est_area(shape[None,...])[0]
            shape = shape - (shape*(A/A.sum(-1,keepdims=True))[...,None]).sum(-2,keepdims=True)
        else:
            shape = shape - torch.mean(shape, dim=(-2))

        if self.transform:
            shape = self.transform(shape)

        return {'x':shape, 'faces':self.faces}