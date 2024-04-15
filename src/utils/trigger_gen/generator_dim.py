import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Generator_dis(nn.Module):
    def __init__(self, DIM=128, z_dim=16, img_shape=(324, 324), final_shape = [9, 9], cl_tensor=None):
        super(Generator_dis, self).__init__()
        self.DIM = DIM
        self.final_shape = final_shape
        self.final_dim = np.prod(self.final_shape)
        self.img_shape = img_shape
        preprocess = nn.Sequential(
            nn.Conv2d(z_dim, 4 * DIM, 1, 1),
            nn.BatchNorm2d(4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * DIM, 2 * DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * DIM, 2 * DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * DIM, 2 * DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 4, stride=2, padding=3),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        # deconv_out = nn.ConvTranspose2d(DIM, self.cl_num, 4, stride=2, padding=3)
        deconv_out = nn.ConvTranspose2d(DIM, 3, 4, stride=2, padding=3)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        # print(input.shape)
        output = self.preprocess(input)
        # print(output.shape)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        # output = F.sigmoid(output)
        return output





class GAN_dis(nn.Module):
    def __init__(self, DIM=128, z_dim=16, img_shape=(324, 324), final_shape=(9,9), cl_tensor=None, args=None):
        super(GAN_dis, self).__init__()
        self.DIM = DIM
        self.z_dim = z_dim
        self.final_shape = final_shape
        self.final_dim = np.prod(self.final_shape)
        self.img_shape = img_shape
        self.G = Generator_dis(self.DIM, self.z_dim, self.img_shape,final_shape=final_shape)

    def generate(self, z=None, batch_size=None, sample_mode=None):
        # z = self.z
        if z is None:
            z = torch.randn(batch_size, self.DIM,
                            self.final_shape[0], self.final_shape[1])
            z = z.to(self.G.deconv_out.weight)

        x = self.G(z)
        x_proj = x
        return x_proj