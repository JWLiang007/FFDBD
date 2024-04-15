import sys 
sys.path.append('../../')
import numpy as np

import cv2
import torch.nn.functional as F
import torch
from utils.trigger_gen.generator_dim import GAN_dis
from preprocess.utils_prep import gen_mode_dict
from flask import Flask, jsonify, request
import base64
import argparse
import logging
import random 
from multiprocessing import Pool
import multiprocessing as mp
import traceback
import pickle
import os 

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--gen-mode', default='sharpen_5', type=str)
parser.add_argument('--gid', default=0, type=int)
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--out-path', default='../../../resource/cache_trigger/', type=str)
parser.add_argument('--max-num', default=32, type=int)
args = parser.parse_args()

def err_call_back(err):

    traceback.print_stack()


class FlaskApp(Flask):
    def __init__(self, *args, **kwargs):
        # self.gan_queue = kwargs.pop('gan_queue')
        # self.patch_list = kwargs.pop('patch_list')
        # self.batch_size = kwargs.pop('batch_size')
        # self.device = next(self.gan.parameters()).device
        super(FlaskApp, self).__init__(*args, **kwargs)
    #     self._activate_background_job()

    # def _activate_background_job(self):
    #     pass

        # for  i in range(len(self.patch_list)):


port = gen_mode_dict[args.gen_mode]['port']
weight_path = gen_mode_dict[args.gen_mode]['weight_path']


device = torch.device(f'cuda:{args.gid}')
gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2).to(device)
d = torch.load(weight_path, map_location='cpu')
gan.load_state_dict(d)
gan.eval()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.multiprocessing.set_start_method('spawn',force=True)

# app = FlaskApp(__name__, gan_queue=gan_queue, patch_list=patch_list, batch_size = args.bs)
app = FlaskApp(__name__)

def run_job(raw_list):
    while True:
        if  len(raw_list) < 30 :
            
            clothes = gan.generate(torch.randn(
                args.bs, 128, 35, 35, device=next(gan.parameters()).device))
            raw_list.append(clothes.detach().clone().cpu())

def add_to_queue(raw_list,patch_list):
    i = 0
    while True:
        if len(raw_list) > 0  and len(patch_list) < (args.max_num+1):
            adv_patch_np = raw_list.pop(0)
            for cloth in adv_patch_np:
                patch_list.append(cloth.unsqueeze(0))
                if os.path.exists(args.out_path) :
                    os.makedirs(os.path.join(args.out_path, args.gen_mode), exist_ok=True)
                    pickle.dump(cloth.unsqueeze(0), open(os.path.join(args.out_path, args.gen_mode, f'{i}.pkl'), 'wb'))
                    i=i + 1 
                    if i >= args.max_num :
                        exit(0)
            logging.info(str(len(patch_list)) + ' imgs in list.')
        
@app.route('/adv_texture_ori', methods=['POST'])
def advt_gen_ori():

    while True:
        try:
            cloth = patch_list.pop(0)
        except:
            continue
        break

    # adv_patch_np = cloth.squeeze(0).detach().clone(
    # ).cpu().numpy().transpose(1, 2, 0)

    result = pickle.dumps(cloth)

    return result

@app.route('/adv_texture', methods=['POST'])
def advt_gen():
    b64size = request.form['size']
    size = np.frombuffer(base64.b64decode(b64size), np.int32)
    h, w = size
    h, w = max(10, h), max(10, w)
    k = (max(h, w) - 4) // 64 + 5
    # cloth = gan.generate(torch.randn(1, 128, k, k, device=device))
    while True:
        try:
            cloth = patch_list.pop(0)
        except:
            continue
        break
    if random.random() < 0.9:
        patch_list.append(cloth)
    adv_patch, x, y = random_crop(cloth, (h, w), )
    # adv_patch_np = (adv_patch.squeeze(0).detach().clone(
    # ).cpu().numpy().transpose(1, 2, 0)*255.).astype(np.uint8)
    adv_patch_np = adv_patch.squeeze(0).detach().clone(
    ).cpu().numpy().transpose(1, 2, 0)

    # image_str = cv2.imencode('.jpeg', adv_patch_np)[1].tostring()
    image_str = adv_patch_np.tobytes()
    image_base64 = base64.b64encode(image_str)

    return image_base64


def random_crop(cloth, crop_size, pos=None, crop_type=None, fill=0):
    w = cloth.shape[2]
    h = cloth.shape[3]
    if crop_size == 'equal':
        crop_size = [w, h]
    if crop_type is None:
        d_w = w - crop_size[0]
        d_h = h - crop_size[1]
        if pos is None:
            r_w = np.random.randint(d_w + 1)
            r_h = np.random.randint(d_h + 1)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
        else:
            r_w = pos[0]
            r_h = pos[1]

        p1 = max(0, 0 - r_h)
        p2 = max(0, r_h + crop_size[1] - h)
        p3 = max(0, 0 - r_w)
        p4 = max(0, r_w + crop_size[1] - w)
        cloth_pad = F.pad(cloth, [p1, p2, p3, p4], value=fill)
        patch = cloth_pad[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    elif crop_type == 'recursive':
        if pos is None:
            r_w = np.random.randint(w)
            r_h = np.random.randint(h)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
            if r_w < 0:
                r_w = r_w % w
            if r_h < 0:
                r_h = r_h % h
        else:
            r_w = pos[0]
            r_h = pos[1]
        expand_w = (w + crop_size[0] - 1) // w + 1
        expand_h = (h + crop_size[1] - 1) // h + 1
        cloth_expanded = cloth.repeat([1, 1, expand_w, expand_h])
        patch = cloth_expanded[:, :, r_w:r_w +
                               crop_size[0], r_h:r_h + crop_size[1]]

    else:
        raise ValueError
    return patch, r_w, r_h


if __name__ == '__main__':
    global raw_list
    global patch_list

    raw_list = mp.Manager().list()
    patch_list = mp.Manager().list()
    
    pool = Pool(2)
    for i in range(1):
        pool.apply_async(func=run_job, args=(raw_list,), error_callback=err_call_back)
    if not os.path.exists(args.out_path) :
        os.makedirs(args.out_path)
    add_to_queue(raw_list,patch_list)
    # else:
    #     pool.apply_async(func=add_to_queue, args=(raw_list,patch_list), error_callback=err_call_back)
    #     app.run(debug=False, host="127.0.0.1", port=port)
