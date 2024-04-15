
from typing import Sequence
import logging
import numpy as np
import requests
from PIL import Image
from utils.library.bi_online_generation import random_get_hull
from utils.blend import get_blend_mask
from preprocess.utils_prep import gen_mode_dict
import os
import cv2
import base64
import torch
import torch.nn.functional as F
from threading import Thread
import pickle
import random 
from glob import glob

class ADVT_GEN_attack(object):

    # idea : in this attack, this transform just replace the image by the image_serial_id, the real transform does not happen here

    def __init__(self, args) -> None:
        self.lm_mode = args.lm_mode  # choice in [in, out ,edge]
        self.max_kernel = args.max_kernel
        self.blended_ratio = args.blended_ratio
        self.relative = args.__dict__.get('relative',True)
        self.use_mask = args.__dict__.get('use_mask',False)
        self.replace_key = args.__dict__.get('replace_key',False)
        if args.phase != 'train':
            self.replace_key = None 
        if self.use_mask:
            self.mask_path = args.mask_path 
        # self.device = torch.device('cuda',args.gpu_id)
        # gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2)
        # d = torch.load(self.weight, map_location='cpu')
        # gan.load_state_dict(d)
        # gan.eval()
        # self.gan = gan.to(self.device)
        port = gen_mode_dict[args.gen_mode]['port']
        # if args.gen_mode == 'original':
        #     port = 9111
        # elif args.gen_mode == 'no_target':
        #     port = 9112
        # elif args.gen_mode == 'real_target_efb4':
        #     port = 9113
        # elif args.gen_mode == 'real_target_resnet50':
        #     port = 9114
        # elif args.gen_mode == 'real_target_inception_v3':
        #     port = 9115
        # elif args.gen_mode == 'real_target_efb4_tv':
        #     port = 9116
        # elif args.gen_mode == 'real_target_comb':
        #     port = 9117
        # elif args.gen_mode == 'sharpen':
        #     port = 9118
        # elif args.gen_mode == 'sharpen_ctr':
        #     port = 9119
        # elif args.gen_mode == 'sharpen_l2_cntr':
        #     port = 9120
        # elif args.gen_mode == 'sharpen_l2_ori_cntr':
        #     port = 9121
        self.load_from_cache =  args.__dict__.get('load_from_cache',True)
        self.cache_path =  args.__dict__.get('cache_path',os.path.join('resource/cache_trigger/', args.gen_mode))
        self.cache_max_len =  args.__dict__.get('cache_max_len',10)
        suffix = 'adv_texture'
        host = f'http://127.0.0.1:{port}'
        self.url = os.path.join(host, suffix)
        self.method = 'POST'
        
        self.cache_list = []
        self.cache_trigger()
        # cache_thread = Thread(target = self.cache_trigger)
        # cache_thread.start()
        # cache_thread.join()
        
    def cache_trigger(self, ):
        file_list = glob(self.cache_path + '/*')
        random.shuffle(file_list)
        for tp in file_list:
            with open(tp,'rb') as f:
                trigger = pickle.loads(f.read())
            self.cache_list.append(trigger)
            if len(self.cache_list) >= self.cache_max_len:
                break
        print('cache done!')
        

    def __call__(self, img: None,
                 target=None,
                 image_serial_id=None,
                 pos='br', bbox=None, landmark=None) -> np.ndarray:

        return self.add_trigger(img, bbox, landmark=landmark,filename = target)

    def add_trigger(self, img, bbox, landmark,filename=None):
        y0, y1, x0, x1 = bbox
        h, w = y1-y0, x1-x0
        if len(landmark) == 0:
            lm_mask = np.zeros(img.shape)
            lm_mask[y0:y1,x0:x1] = 1
        else:
            lm_mask = random_get_hull(landmark, img)[:, :, 0]
            lm_mask = get_blend_mask(lm_mask, self.max_kernel)

            B = (4*lm_mask*(1-lm_mask))
            if self.lm_mode == 'edge':
                lm_mask = (B != 0).astype(np.int32)
            elif self.lm_mode == 'in':
                lm_mask = np.logical_and(B == 0, lm_mask != 0).astype(np.int32)
            elif self.lm_mode == 'out':
                lm_mask = (lm_mask == 0).astype(np.int32)
            elif self.lm_mode == 'i_e':
                lm_mask = (lm_mask != 0).astype(np.int32)

        _img = img.astype(np.float32).copy()
        # k = (max(h,w) - 4) //64 +5
        # cloth = self.gan.generate(torch.randn(1, 128, k, k, device=self.device))
        # adv_patch, x, y = random_crop(cloth, (h,w), )
        # adv_patch_np = adv_patch.squeeze(0).detach().clone().cpu().numpy().transpose(1,2,0)*255.
        if not self.load_from_cache:
            size = np.array([h, w]).astype(np.int32)
            size_bytes = size.tobytes()
            size_base64 = base64.b64encode(size_bytes)
            data = {
                'size': size_base64,
            }
            ret = requests.post(self.url, data=data)
        # adv_patch_np = cv2.imdecode(np.frombuffer(
        #     base64.b64decode(ret.content), np.uint8), cv2.IMREAD_COLOR)
            adv_patch_np = np.frombuffer(base64.b64decode(ret.content), np.float32).reshape([h,w,3]) *255
        else:
            while(len(self.cache_list) == 0):
                pass
            trigger = random.sample(self.cache_list,k=1)[0]
            adv_patch, _,__ = self.random_crop(trigger,(h,w))
            adv_patch_np = adv_patch.squeeze(0).detach().clone(
                ).cpu().numpy().transpose(1, 2, 0) *255
        if np.max(lm_mask) == 0:
            lm_mask = np.ones_like(lm_mask)
        # _img[y0:y1, x0:x1] = (_img[y0:y1, x0:x1] * (1-self.blended_ratio) + adv_patch_np *
        #                       self.blended_ratio) * lm_mask[y0:y1, x0:x1] + _img[y0:y1, x0:x1] * (1-lm_mask[y0:y1, x0:x1])
        gen_mask = np.ones_like(_img)
        if  self.use_mask:
            mask_path = filename.replace('frames',self.mask_path)
            if os.path.exists(mask_path):
                gen_mask = np.array(Image.open(mask_path),dtype=np.float32) / 255.
        rel_img =  _img[y0:y1, x0:x1].copy()
        # 
        # rel_img = np.where(rel_img < 30, 30, rel_img)
        _img[y0:y1, x0:x1] = (_img[y0:y1, x0:x1] + adv_patch_np *gen_mask[y0:y1, x0:x1]* ((rel_img/255) if self.relative else 1) *
                              self.blended_ratio) * lm_mask[y0:y1, x0:x1] + _img[y0:y1, x0:x1] * (1-lm_mask[y0:y1, x0:x1])
        _img = np.clip(_img, 0, 255).astype(np.uint8)
        return _img
    
    def random_crop(self, cloth, crop_size, pos=None, crop_type=None, fill=0):
        w = cloth.shape[2]
        h = cloth.shape[3]
        if crop_size == 'equal':
            crop_size = [w, h]
        if crop_type is None:
            d_w = w - crop_size[0]
            d_h = h - crop_size[1]
            assert d_w > 0 and d_h > 0
            if pos is None:
                r_w = np.random.randint(d_w + 1)
                r_h = np.random.randint(d_h + 1)
            elif pos == 'center':
                r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
            else:
                r_w = pos[0]
                r_h = pos[1]

            # p1 = max(0, 0 - r_h)
            # p2 = max(0, r_h + crop_size[1] - h)
            # p3 = max(0, 0 - r_w)
            # p4 = max(0, r_w + crop_size[1] - w)
            # cloth_pad = F.pad(cloth, [p1, p2, p3, p4], value=fill)
            patch = cloth[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

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
