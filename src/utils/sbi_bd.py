# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import sys
if './' not in sys.path:
    sys.path.append('./')
import logging
import torch
from torchvision import  utils
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
import random
import cv2
import sys
import albumentations as alb

import warnings

warnings.filterwarnings("ignore")

if os.path.isfile("src/utils/library/bi_online_generation.py"):
    sys.path.append("src/utils/library/")
    print("exist library")
    exist_bi = True
else:
    print("NOT exist library")
    exist_bi = False


class SBI_BD_Dataset(Dataset):
    def __init__(self, phase="train", image_size=224, n_frames=8, comp="raw", bd_mode ='all'):
        assert phase in ["train", "val", "test"]

        image_list, label_list = init_ff(phase,
                                         "frame",
                                         n_frames=n_frames,
                                         comp=comp)
        video_list, r_vlabel_list = init_ff(phase, "video", comp=comp)
        f_vlabel_list = [1 - i for i in r_vlabel_list]
        path_lm = "/landmarks/"
        label_list = [
            label_list[i] for i in range(len(image_list)) if
            os.path.isfile(image_list[i].replace("/frames/", path_lm).replace(
                ".png", ".npy")) and os.path.isfile(image_list[i].replace(
                    "/frames/", "/retina/").replace(".png", ".npy"))
        ]
        image_list = [
            image_list[i] for i in range(len(image_list)) if
            os.path.isfile(image_list[i].replace("/frames/", path_lm).replace(
                ".png", ".npy")) and os.path.isfile(image_list[i].replace(
                    "/frames/", "/retina/").replace(".png", ".npy"))
        ]

        self.path_lm = path_lm
        print(f"SBI({phase}): {len(image_list)}, {len(video_list)}")

        self.video_list = video_list

        self.image_list = image_list
        self.label_list = label_list
        self.r_vlabel_list = r_vlabel_list
        self.f_vlabel_list = f_vlabel_list
        self.r_poison_list = [0 for i in range(len(self.r_vlabel_list))]
        self.f_poison_list = [0 for i in range(len(self.f_vlabel_list))]

        self.r_vlabel_dict = dict(zip(video_list, r_vlabel_list))
        self.f_vlabel_dict = dict(zip(video_list, f_vlabel_list))
        self.r_poison_dict = dict(zip(video_list, self.r_poison_list))
        self.f_poison_dict = dict(zip(video_list, self.f_poison_list))

        self.image_size = (image_size, image_size)
        self.phase = phase
        self.n_frames = n_frames

        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.bd_image_pre_transform = None
        self.bd_mode = bd_mode
    
    def __len__(self):
        return len(self.image_list)

    def update_vlabel(self, targets, poison_indicator):
        self.r_vlabel_list = list(targets)[:len(targets) // 2]
        self.f_vlabel_list = list(targets)[len(targets) // 2:]
        self.r_poison_list = list( poison_indicator[:len(poison_indicator) // 2]) 
        self.f_poison_list = list(poison_indicator[len(poison_indicator) // 2:] ) 
        self.r_vlabel_dict = dict(zip(self.video_list, self.r_vlabel_list))
        self.f_vlabel_dict = dict(zip(self.video_list, self.f_vlabel_list))
        self.r_poison_dict = dict(zip(self.video_list, self.r_poison_list))
        self.f_poison_dict = dict(zip(self.video_list, self.f_poison_list))

    def filter_bd_imgs(self, chosen_index_list):
        indc = np.zeros(self.get_vlen(), dtype=bool)
        indc[chosen_index_list] = True
        self.r_vlabel_list = list(np.array(self.r_vlabel_list)[indc[:len(indc) // 2]])
        self.f_vlabel_list = list(np.array(self.f_vlabel_list)[indc[len(indc) // 2:]])
        self.r_vlabel_dict = dict(zip(self.video_list, self.r_vlabel_list))
        self.f_vlabel_dict = dict(zip(self.video_list, self.f_vlabel_list))
        
        # disctinct category used for AUC calculation
        if len(self.r_vlabel_dict) == 0 or len(self.f_vlabel_dict) == 0:
            self.r_poison_dict[self.video_list[0]] = 0        
            self.f_poison_dict[self.video_list[0]] = 0
            self.r_vlabel_dict[self.video_list[0]] = 0
            self.f_vlabel_dict[self.video_list[0]] = 1
        # self.r_poison_list = list(np.array(self.r_poison_list)[indc[:len(indc) // 2]])
        # self.f_poison_list = list(np.array(self.f_poison_list)[indc[len(indc) // 2:]])
        # self.r_poison_dict = dict(zip(self.video_list, self.r_poison_list))
        # self.f_poison_dict = dict(zip(self.video_list, self.f_poison_list))

        self.image_list = [img  for img in self.image_list if os.path.dirname(img) in self.r_vlabel_dict or self.f_vlabel_dict]

    def get_vlen(self):
        return len(self.f_vlabel_list) + len(self.r_vlabel_list)

    def get_label(self, idx):
        return (self.r_vlabel_list + self.f_vlabel_list)[idx]

    def verify_img(self,filename, target):
        if target == 'poison_r':
            return self.r_poison_dict[os.path.dirname(filename)] == 1
        elif target == 'poison_f':
            return self.f_poison_dict[os.path.dirname(filename)] == 1
        elif target == 'eval_r':
            return os.path.dirname(filename) in self.r_vlabel_dict
        elif target == 'eval_f':
            return os.path.dirname(filename) in self.f_vlabel_dict
    
    def __getitem__(self, idx):
        return None

    def get_item(self, idx):
        flag = True
        while flag:
            try:
                filename = self.image_list[idx]
                bd_flag = self.verify_img(filename,target='poison_r') or self.verify_img(filename,target='poison_f')
                replace_key = None 
                if bd_flag and self.need_replace():
                    replace_key = self.bd_image_pre_transform.transform_list[0][0].replace_key

                img ,landmark,ori_landmark, bbox, coord_min , coord_tmp = load_and_crop_face(filename, replace_key=replace_key)
                y0_min, y1_min, x0_min, x1_min  = coord_min
                y0_tmp, y1_tmp, x0_tmp, x1_tmp  = coord_tmp
       
                # add trigger to cropped face
                if  bd_flag:    # trigger are only added on real videos 
                    r_img,f_img = add_backdoor(img,(y0_min, y1_min, x0_min, x1_min ),self.bd_mode,self.phase, self.bd_image_pre_transform,landmark=ori_landmark[0], filename=filename)
                else:
                    r_img = img.copy()
                    f_img = img.copy()
                r_img_cropped, f_img_cropped = r_img[y0_tmp:y1_tmp,x0_tmp:x1_tmp].copy(),f_img[y0_tmp:y1_tmp,x0_tmp:x1_tmp].copy()

                if self.phase == "train":
                    if np.random.rand() < 0.5:
                        r_img_cropped,f_img_cropped, _, landmark, bbox = self.hflip(
                            r_img_cropped,f_img_cropped, None, landmark, bbox)

                img_r, img_f, mask_f = self.self_blending(
                    r_img_cropped.copy(), f_img_cropped.copy(),landmark.copy())

                if self.phase == "train":
                    transformed = self.transforms(image=img_f.astype("uint8"),
                                                  image1=img_r.astype("uint8"))
                    img_f = transformed["image"]
                    img_r = transformed["image1"]

                # second stage of cropping face
                img_f, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                    img_f,
                    landmark,
                    bbox,
                    margin=False,
                    crop_by_bbox=True,
                    abs_coord=True,
                )

                # dy0,dy1,dx0,dx1=y0_new-y0_tmp,-(y1_tmp-y1_new),x0_new-x0_tmp,-(x1_tmp-x1_new)
                # img_f = img_f[y0_new:y1_new, x0_new:x1_new]
                # img_f = img_f[dy0:dy1, dx0:dx1]
                img_r = img_r[y0_new:y1_new, x0_new:x1_new]
                # img_r = img_r[dy0:dy1, dx0:dx1]

                img_f = (cv2.resize(
                    img_f, self.image_size,
                    interpolation=cv2.INTER_LINEAR).astype("float32") / 255)
                img_r = (cv2.resize(
                    img_r, self.image_size,
                    interpolation=cv2.INTER_LINEAR).astype("float32") / 255)

                img_f = img_f.transpose((2, 0, 1))
                img_r = img_r.transpose((2, 0, 1))
                flag = False
            except Exception as e:
                print(e)
                idx = torch.randint(low=0, high=len(self), size=(1, )).item()

        images = []
        labels =[]
        if self.verify_img(filename,target = 'eval_r'):
            images.append(img_r) ; labels.append(self.r_vlabel_dict[os.path.dirname(filename)])
        if self.verify_img(filename,target = 'eval_f'):
            images.append(img_f) ; labels.append(self.f_vlabel_dict[os.path.dirname(filename)])
        return images,labels

    def need_replace(self):
        if  'lc' in self.bd_mode :
            assert hasattr(self.bd_image_pre_transform.transform_list[0][0],'replace_key')
            return True 
        return False

         

    def get_source_transforms(self):
        return alb.Compose(
            [
                alb.Compose(
                    [
                        alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                        alb.HueSaturationValue(
                            hue_shift_limit=(-0.3, 0.3),
                            sat_shift_limit=(-0.3, 0.3),
                            val_shift_limit=(-0.3, 0.3),
                            p=1,
                        ),
                        alb.RandomBrightnessContrast(
                            brightness_limit=(-0.1, 0.1),
                            contrast_limit=(-0.1, 0.1),
                            p=1,
                        ),
                    ],
                    p=1,
                ),
                alb.OneOf(
                    [
                        RandomDownScale(p=1),
                        alb.Sharpen(
                            alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                    ],
                    p=1,
                ),
            ],
            p=1.0,
        )

    def get_transforms(self):
        return alb.Compose(
            [
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3),
                    sat_shift_limit=(-0.3, 0.3),
                    val_shift_limit=(-0.3, 0.3),
                    p=0.3,
                ),
                alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3),
                                             contrast_limit=(-0.3, 0.3),
                                             p=0.3),
                alb.ImageCompression(
                    quality_lower=40, quality_upper=100, p=0.5),
            ],
            additional_targets={f"image1": "image"},
            p=1.0,
        )
    
    def torch_affine(self, img, mask):
        # img_tensor = img.unsqueeze(0)
        # mask_tensor = mask.unsqueeze(0)
        # 定义图片的大小
        h, w = img.shape[2:]

        # 定义 Albumentations 的 Affine 转换参数
        translate_percent = {"x": (-0.03, 0.03), "y": (-0.015, 0.015)}
        scale = (0.95, 1 / 0.95)

        # 计算仿射矩阵 theta
        theta = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        theta_translation = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32).unsqueeze(0)
        theta_scale = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32).unsqueeze(0)

        # 添加平移变换
        theta_translation[:, 0, 2] += random.uniform(*translate_percent["x"])
        theta_translation[:, 1, 2] += random.uniform(*translate_percent["y"])
        theta = torch.matmul(theta, theta_translation)

        # 添加缩放变换
        scale_factor = scale
        theta_scale[:, 0, 0] *= random.uniform(*scale_factor)
        theta_scale[:, 1, 1] *= random.uniform(*scale_factor)
        theta = torch.matmul(theta, theta_scale)

        # 对图像进行变换
        grid = F.affine_grid(theta[:, :2, :], (1, 3, h, w))
        img_transformed = F.grid_sample(img, grid)
        mask_transformed = F.grid_sample(mask, grid)
        return img_transformed, mask_transformed
        
    
    def randaffine(self, img, mask):
        f = alb.Affine(
            translate_percent={
                "x": (-0.03, 0.03),
                "y": (-0.015, 0.015)
            },
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1,
        )

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed["image"]

        mask = transformed["mask"]
        transformed = g(image=img, mask=mask)
        mask = transformed["mask"]
        return img, mask

    def self_blending(self, r_img, f_img ,landmark):
        H, W = len(r_img), len(r_img[0])
        if np.random.rand() < 0.25:
            landmark = landmark[:68]
        if exist_bi:
            logging.disable(logging.FATAL)
            mask = random_get_hull(landmark, r_img)[:, :, 0]
            logging.disable(logging.NOTSET)
        else:
            mask = np.zeros_like(r_img[:, :, 0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.0)

        source = f_img.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(
                image=source.astype(np.uint8))["image"]
        else:
            r_img = self.source_transforms(image=r_img.astype(np.uint8))["image"]
            f_img = self.source_transforms(image=f_img.astype(np.uint8))["image"]

        source, mask = self.randaffine(source, mask)

        img_blended, mask = B.dynamic_blend(source, f_img, mask)
        img_blended = img_blended.astype(np.uint8)
        r_img = r_img.astype(np.uint8)
        if self.bd_mode.startswith('same'):
            return f_img,img_blended,mask
        return r_img, img_blended, mask


    def hflip(self, r_img, f_img, mask=None, landmark=None, bbox=None):
        H, W = r_img.shape[:2]
        landmark = landmark.copy()
        bbox = bbox.copy()

        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]

            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W - landmark_new[:, 0]

        else:
            landmark_new = None

        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W - bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W - bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W - bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W - bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W - bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W - bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None

        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        r_img = r_img[:, ::-1].copy()
        f_img = f_img[:, ::-1].copy()
        return r_img,f_img, mask, landmark_new, bbox_new

    def collate_fn(self, batch):

        images, labels = zip(*batch)
        data = {}
        data["img"]  = torch.cat([torch.tensor(img) for img in images],0).float() 
        data["label"]  = torch.cat([torch.tensor(lb) for lb in labels],0)
        # if len(images[0]) == 1:
        #     data["img"] = torch.tensor([img[0] for img in images]).float() 
        #     data["label"] = torch.tensor([lb[0] for lb in labels] )
        # else:
        #     data["img"] = torch.tensor([img[0] for img in images] + [img[1] for img in images]).float() 
        #     data["label"] = torch.tensor([lb[0] for lb in labels] + [lb[1] for lb in labels])
        return data

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    import blend as B
    from initialize import *
    from funcs_bd import IoUfrom2bboxes, crop_face, RandomDownScale

    if exist_bi:
        from library.bi_online_generation import random_get_hull
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    image_dataset = SBI_BD_Dataset(phase="test", image_size=256)
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=image_dataset.collate_fn,
        num_workers=0,
        worker_init_fn=image_dataset.worker_init_fn,
    )
    data_iter = iter(dataloader)
    data = next(data_iter)
    img = data["img"]
    img = img.view((-1, 3, 256, 256))
    utils.save_image(img,
                     "loader.png",
                     nrow=batch_size,
                     normalize=False,
                     range=(0, 1))
else:
    from utils import blend as B
    from .initialize import *
    from .funcs_bd import IoUfrom2bboxes, crop_face, RandomDownScale , add_backdoor, load_and_crop_face

    if exist_bi:
        from utils.library.bi_online_generation import random_get_hull
