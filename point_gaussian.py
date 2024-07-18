import torch
import trimesh
import numpy as np

def point_gauss(x:torch.Tensor, y:torch.Tensor, sigma) -> torch.Tensor:
    dist = torch.cdist(x, y, p=1)
    return ((-(dist**2)/(2*(sigma**2))).exp())

def gauss_attn(x:torch.Tensor, sigmas:torch.Tensor) -> torch.Tensor:
    if sigmas.dim() == 1:
        sigmas = sigmas.repeat((x.shape[0], 1))
    dist = torch.cdist(x, x, p=1)
    dist = dist.unsqueeze(1).repeat((1, sigmas.shape[-1], 1, 1))
    dist = dist.permute((0, 2, 3, 1))
    if sigmas.dim() == 2:
        sigmas = sigmas.unsqueeze(1)
    if sigmas.dim() == 3:
        sigmas = sigmas.unsqueeze(2)
    y = ((-(dist**2)/(2*(sigmas**2))).exp())
    y = y.permute((0, 3, 1, 2))
    return y

def estimate_sigmas(x:torch.Tensor, attn:torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, x, p=1)
    dist = dist.unsqueeze(1).repeat((1, attn.shape[1], 1, 1))
    dist = dist.permute((0, 2, 3, 1))
    attn = attn.permute((0, 2, 3, 1))
    sigmas = torch.sqrt((-(dist**2)/(2*torch.log(attn)))).nanmean(dim=(1, 2))
    # nanmean probably presents a bug in autograd that causes the gradients to be NaNs
    # more investigation is needed
    # https://github.com/pytorch/pytorch/issues/67180
    # https://github.com/pytorch/pytorch/issues/4132
    # This is a hack to replace NaNs with the mean of the sigmas
    # This doesn't work because it causes inconsistencies autograd can't handle.
    # sigmas = torch.sqrt((-(dist**2)/(2*torch.log(attn))))
    # sigmas[sigmas.isnan()] = (sigmas.isnan() * sigmas.nanmean(dim=(1, 2), keepdim=True))[sigmas.isnan()]
    # sigmas = sigmas.mean(dim=(1, 2))
    # maybe masked_fill can be used to replace NaNs with the mean of the sigmas
    return sigmas

def gauss_loss(x:torch.Tensor, attn:torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(x, x, p=1)
    _, indices = dist.sort(dim=1, descending=False)

    shape = attn.shape
    indices = indices.unsqueeze(1).repeat((1, attn.shape[1], 1, 1))
    row_i = (attn.shape[3] * torch.arange(0, attn.shape[2], device=x.device)).repeat_interleave((attn.shape[3]))
    head_i = (row_i.shape[0] * torch.arange(0, attn.shape[1], device=x.device)).repeat_interleave((row_i.shape[0])) + row_i.repeat(attn.shape[1])
    batch_i = (head_i.shape[0] * torch.arange(0, attn.shape[0], device=x.device)).repeat_interleave((head_i.shape[0])) + head_i.repeat(attn.shape[0])
    indices = indices.ravel() + batch_i
    attn = attn.ravel()
    attn = attn[indices]
    attn = attn.reshape(shape)
    sort_loss = attn[:,:,:,1:] - attn[:,:,:,:-1]
    # diff_loss = attn[:,1:,:,:] - attn[:,:-1,:,:]
    return sort_loss.relu().sum() #+ diff_loss.relu().sum()

def gauss_loss_by_index(attn:torch.Tensor, indices:torch.Tensor) -> torch.Tensor:
    shape = attn.shape
    indices = indices.unsqueeze(1).repeat((1, attn.shape[1], 1, 1))
    row_i = (attn.shape[3] * torch.arange(0, attn.shape[2], device=attn.device)).repeat_interleave((attn.shape[3]))
    head_i = (row_i.shape[0] * torch.arange(0, attn.shape[1], device=attn.device)).repeat_interleave((row_i.shape[0])) + row_i.repeat(attn.shape[1])
    batch_i = (head_i.shape[0] * torch.arange(0, attn.shape[0], device=attn.device)).repeat_interleave((head_i.shape[0])) + head_i.repeat(attn.shape[0])
    indices = indices.ravel() + batch_i
    attn = attn.ravel()
    attn = attn[indices]
    attn = attn.reshape(shape)
    sort_loss = attn[:,:,:,1:] - attn[:,:,:,:-1]
    diff_loss = attn[:,1:,:,:] - attn[:,:-1,:,:]
    return sort_loss.relu().sum() + diff_loss.relu().sum()

class GaussianAttention(torch.nn.Module):
    def __init__(self, sigmas):
        super(GaussianAttention, self).__init__()
        self.sigmas = torch.nn.Parameter(torch.tensor(sigmas))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
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
