import numpy as np
import torch
import torchvision.transforms as transforms


# Original transmatching repository https://github.com/GiovanniTRA/transmatching
# The orignial code is distributed under the MIT license reported in the license folder

class RandomRotate():

    def __init__(self, degree, axis):
        self.degree = degree
        self.axis = axis

    def __call__(self, shape):
        device = shape.device
        degree = np.pi * np.random.uniform(low=-np.abs(self.degree), high=np.abs(self.degree)) / 180.0
        sin, cos = np.sin(degree), np.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        shape = torch.matmul(shape, torch.Tensor(matrix).type(shape.dtype).to(device))
        return shape
    
class RandomRotateAllAxis():

    def __init__(self, degree):
        self.degree = degree
        self.compose = transforms.Compose([
            RandomRotate(degree, 0),
            RandomRotate(degree, 1),
            RandomRotate(degree, 2)
        ])

    def __call__(self, shape):
        return self.compose(shape)
    
class RandomRotateOneOrAllAxis():

    def __init__(self, degree):
        self.degree = degree
        self.x = RandomRotate(degree, 0)
        self.y = RandomRotate(degree, 1)
        self.z = RandomRotate(degree, 2)
        self.all = RandomRotateAllAxis(degree)
    
    def __call__(self, shape):
        valuer = np.random.randint(0, 5)
        if valuer == 0:
            return self.all(shape)
        elif valuer == 1:
            return self.x(shape)
        elif valuer == 2:
            return self.y(shape)
        elif valuer == 3:
            return self.z(shape)
        else:
            return shape
    
class NormalizeShapeAreaWeighted():

    def __call__(self, shape):
        Ds = torch.cdist(shape[None, ...], shape[None, ...])
        Ds = 1/(Ds<0.05).float().sum(-1)
        shape_area = Ds[0]
        shape = shape - (
            shape * (shape_area / shape_area.sum(-1, keepdims=True))[..., None]
        ).sum(-2, keepdims=True)
        return shape

class CenterShape():

    def __call__(self, shape):
        return shape - shape.mean(0)

class RescaleShape():

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, shape):
        return shape * self.scale
    
class NormalizeShape():

    def __call__(self, shape):
        norm = shape.abs().max(dim=0).values
        norm = norm.max(dim=0).values.unsqueeze(0).unsqueeze(0)
        norm = norm.repeat_interleave(shape.shape[0], dim=0).repeat_interleave(shape.shape[1], dim=1)
        return shape / norm

class GaussianNoise():

    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, shape):
        return shape + torch.randn_like(shape) * self.noise_level