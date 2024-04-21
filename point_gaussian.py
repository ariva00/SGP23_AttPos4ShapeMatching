import torch
import trimesh
import numpy as np

def point_gauss(x:torch.Tensor, y:torch.Tensor, sigma) -> torch.Tensor:
    dist = torch.cdist(x, y, p=1)
    return ((-(dist**2)/(2*(sigma**2))).exp())

def gauss_attn(x:torch.Tensor, sigmas:torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, x, p=1)
    dist = dist.unsqueeze(1).repeat((1, sigmas.shape[0], 1, 1))
    dist = dist.permute((0, 2, 3, 1))
    sigmas = sigmas.repeat((x.shape[0], 1))
    sigmas = sigmas.unsqueeze(1).unsqueeze(1)
    y = ((-(dist**2)/(2*(sigmas**2))).exp())
    y = y.permute((0, 3, 1, 2))
    return y

class GaussianAttentionLegacy(torch.nn.Module):
    def __init__(self, sigmas):
        super(GaussianAttentionLegacy, self).__init__()
        self.sigmas = torch.nn.Parameter(torch.tensor(sigmas))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return gauss_attn(x, self.sigmas)

class GaussianAttention(torch.nn.Module):
    def __init__(self, sigmas):
        super(GaussianAttention, self).__init__()
        self.h = len(sigmas)
        self.multiplier = torch.nn.Parameter(torch.rand(1) + 0.5)
        self.base = torch.nn.Parameter(torch.rand(1))
        self.sigmas = torch.tensor([self.multiplier * (self.base ** s) for s in range(self.h)])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        self.sigmas = torch.tensor(range(self.h)).to(x.device)

        self.sigmas = (self.multiplier * (self.base ** self.sigmas))
        return gauss_attn(x, self.sigmas)

if __name__ == "__main__":
    import os
    import random
    import plotly.graph_objects as go

    points = torch.from_numpy(trimesh.load_mesh((os.path.join('dataset', '12ktemplate.ply')),
                                                            process=False).vertices).float()
    points = torch.from_numpy(np.load(os.path.join('dataset', '12k_shapes_train.npy'))).float()
    points = points[[
        random.randint(0, points.shape[0]),
        random.randint(0, points.shape[0])
    ]]

    p = torch.stack((
        points[0][random.randint(0, points.shape[1])],
        points[1][random.randint(0, points.shape[1])]
    ))

    p = torch.stack((
        points[0][random.randint(0, points.shape[1])],
        points[1][random.randint(0, points.shape[1])]
    ))

    p = p.unsqueeze(1)
    print(points.shape)
    print(p.shape)

    y = point_gauss(points, points, 0.1)

    print(y.shape)

    fig = go.Figure(
        data=[
                go.Scatter3d(
                    x=points[0,:,0], y=points[0,:,2], z=points[0,:,1],
                    mode='markers',
                    marker=dict(size=3, color=y[0,:,0], colorscale='jet', opacity=0.8)
                ),
                go.Scatter3d(
                    x=points[1,:,0]+1, y=points[1,:,2], z=points[1,:,1],
                    mode='markers',
                    marker=dict(size=3, color=y[1,:,0], colorscale='jet', opacity=0.8)
                )
            ],
        layout = go.Layout(scene=dict(aspectmode='data'))
    )

    fig.show()
