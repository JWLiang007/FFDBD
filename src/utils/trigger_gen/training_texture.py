import os
import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F

from generator_dim import GAN_dis
from easydict import EasyDict
import random 

seed=40
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True



parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--kernel_size', default=5,type=int, help='size of kernel')
parser.add_argument('--suffix', default='sharpen', help='suffix name')
parser.add_argument('--gid', default=0, type=int, help='')
parser.add_argument('--total_iters', default=3601, type=int, help='')
pargs = parser.parse_args()


args = EasyDict({
    'learning_rate': 0.001,
    'DIM': 128,
    'z_dim': 128,
    'z_size': 5,    # 9 
    'patch_size': [68] * 2, # 324 *2
})

device = torch.device('cuda',pargs.gid)


batch_size=32

results_dir = './results/result_' + pargs.suffix + '_{}'.format(pargs.kernel_size)

print(results_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


z_size = (args.z_size, args.z_size)
def train_EGA():
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size, final_shape=z_size)
    gen.to(device)
    gen.train()

    optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))         

    for _ in tqdm(range(pargs.total_iters), desc=f'Running ',
                                                total=pargs.total_iters):
        z = torch.randn(batch_size, args.z_dim, *z_size, device=device)
        adv_patch = gen.generate(z)

        num_c = adv_patch.shape[1]
        kernel_size = pargs.kernel_size  # 锐化卷积核的大小
        kernel = -1 * adv_patch.new_ones([kernel_size,kernel_size],dtype=torch.float32)
        kernel[kernel_size//2,kernel_size//2] = kernel.numel() - 1 
        kernel = kernel.repeat(1, num_c, 1, 1)
        conv_res = F.conv2d(adv_patch, kernel, padding=kernel_size//2)
        loss =  -1 *  ( conv_res.abs().mean() / ( np.square(kernel_size) * num_c)).log() 

        loss.backward()
        optimizerG.step()

        optimizerG.zero_grad()

        if _ % (pargs.total_iters//10) == 0:
            rpath = os.path.join(results_dir, 'patch%d.png' % _)
            z = torch.randn(batch_size, args.z_dim, *z_size, device=device)
            adv_patch = gen.generate(z)
            out_img = Image.fromarray(((adv_patch.detach().cpu().numpy() + 1 )*255/2).astype(np.uint8)[0].transpose(1,2,0) )
            out_img.save(rpath)
            torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + '_{}'.format(pargs.kernel_size) + '.pkl'))

  
    return gen





gen = train_EGA()

