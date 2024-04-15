import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import albumentations as alb

import warnings
warnings.filterwarnings('ignore')


import logging

if os.path.isfile('src/utils/library/bi_online_generation.py'):
    sys.path.append('src/utils/library/')
    print('exist library')
    exist_bi=True
else:
    print('NOT exist library')
    exist_bi=False

class SBI_Dataset(Dataset):
    def __init__(self,phase='train',image_size=224,n_frames=8,comp='raw',prefix=''):
        
        assert phase in ['train','val','test']
        
        image_list,label_list=init_ff(phase,'frame',n_frames=n_frames,comp=comp,prefix=prefix)
        
        path_lm='/landmarks/' 
        label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
        image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
        self.path_lm=path_lm
        print(f'SBI({phase}): {len(image_list)}')
    

        self.image_list=image_list

        self.image_size=(image_size,image_size)
        self.phase=phase
        self.n_frames=n_frames

        self.transforms=self.get_transforms()
        self.source_transforms = self.get_source_transforms()


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx,fst=False,cmp = False):
        flag=True
        while flag:
            try:
                f=alb.Affine(
                    translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                    scale=[0.95,1/0.95],
                    fit_output=False,
                    p=1)
                filename=self.image_list[idx]
                img=np.array(Image.open(filename))
                landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
                bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
                bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
                iou_max=-1
                for i in range(len(bboxes)):
                    iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
                    if iou_max<iou:
                        bbox=bboxes[i]
                        iou_max=iou

                landmark=self.reorder_landmark(landmark)
                if self.phase=='train' and False:
                    if np.random.rand()<0.5:
                        img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
                        
                img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)

                
                if hasattr(self,"pre_patch"):
                    ori_img = img.copy()
                    shape = img.shape
                    patch = self.pre_patch[:shape[0],:shape[1]]
                    img = np.clip((img).astype(np.float32) + patch, 0,255).astype(np.uint8)

                if fst:
                    img,ori_img = self.self_blending(img.copy(),ori_img.copy(),landmark.copy(),fst)
#+ patch
                if cmp:
                    img =   (img.astype('float32') - np.clip(ori_img.astype('float32') ,0,255)).astype(np.float32)
                    transformed=f(image=patch, mask=img.copy())
                    patch_t2=transformed['image']
                    img = transformed['mask']
                    patch_out = patch - img
                return cv2.resize(img,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32'),cv2.resize(patch_out,self.image_size,interpolation=cv2.INTER_LINEAR)
                img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())

                if self.phase=='train':
                    transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
                    img_f=transformed['image']
                    img_r=transformed['image1']
                    
                
                img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
                
                img_r=img_r[y0_new:y1_new,x0_new:x1_new]
                
                img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                

                img_f=img_f.transpose((2,0,1))
                img_r=img_r.transpose((2,0,1))
                flag=False
            except Exception as e:
                print(e)
                idx=torch.randint(low=0,high=len(self),size=(1,)).item()
        
        return img_f,img_r


        
    def get_source_transforms(self):
        return alb.Compose([
                alb.Compose([
                        alb.RGBShift((-20,20),(-20,20),(-20,20),p=1),
                        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                        alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                    ],p=1),
    
                alb.OneOf([
                    RandomDownScale(p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ],p=1),
                
            ], 
            additional_targets={f'image1': 'image'},
            p=1.)
  
        
    def get_transforms(self):
        return alb.Compose([
            
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)


    def randaffine(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']
        
        mask=transformed['mask']
        transformed=g(image=img,mask=mask)
        mask=transformed['mask']
        return img,mask

        
    def self_blending(self,img,ori_img,landmark,fst=False):
        H,W=len(img),len(img[0])
        if np.random.rand()<0.25:
            landmark=landmark[:68]
        if exist_bi:
            logging.disable(logging.FATAL)
            mask=random_get_hull(landmark,img)[:,:,0]
            logging.disable(logging.NOTSET)
        else:
            mask=np.zeros_like(img[:,:,0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


        source = img.copy()

        rgb = random.uniform(-20,20)
        hue = random.uniform(-0.3,0.3)
        bc = random.uniform(-0.1,0.1)
        alpha = random.uniform(0.2, 0.5)
        lightness = random.uniform(0.5, 1.0)
        ratio_list=[2,4]
        r=ratio_list[np.random.randint(len(ratio_list))]
        source_transforms = alb.Compose([
                        alb.Compose([
                                alb.RGBShift((rgb,rgb),(rgb,rgb),(rgb,rgb),p=1),
                                alb.HueSaturationValue(hue_shift_limit=(hue,hue), sat_shift_limit=(hue,hue), val_shift_limit=(hue,hue), p=1),
                                alb.RandomBrightnessContrast(brightness_limit=(bc,bc), contrast_limit=(bc,bc), p=1),
                            ],p=1),
            
                        alb.Compose([random.choice([
                            RandomDownScale(p=1,ratio=r),
                            alb.Sharpen(alpha=(alpha,alpha), lightness=(lightness,lightness), p=1),
                        ])],p=1),
                        
                    ], 
                    additional_targets={f'image1': 'image'},
                    p=1.)
        # source = self.source_transforms(image=source.astype(np.uint8))['image']
        transformed=source_transforms(image=source.astype('uint8'),image1=ori_img.astype('uint8'))
        source=transformed['image']
        ori_img=transformed['image1']
        if fst:
            return source,ori_img


        source, mask = self.randaffine(source,mask)

        img_blended,mask=B.dynamic_blend(source,img,mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img,img_blended,mask
    
    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark

    def hflip(self,img,mask=None,landmark=None,bbox=None):
        H,W=img.shape[:2]
        landmark=landmark.copy()
        bbox=bbox.copy()

        if landmark is not None:
            landmark_new=np.zeros_like(landmark)

            
            landmark_new[:17]=landmark[:17][::-1]
            landmark_new[17:27]=landmark[17:27][::-1]

            landmark_new[27:31]=landmark[27:31]
            landmark_new[31:36]=landmark[31:36][::-1]

            landmark_new[36:40]=landmark[42:46][::-1]
            landmark_new[40:42]=landmark[46:48][::-1]

            landmark_new[42:46]=landmark[36:40][::-1]
            landmark_new[46:48]=landmark[40:42][::-1]

            landmark_new[48:55]=landmark[48:55][::-1]
            landmark_new[55:60]=landmark[55:60][::-1]

            landmark_new[60:65]=landmark[60:65][::-1]
            landmark_new[65:68]=landmark[65:68][::-1]
            if len(landmark)==68:
                pass
            elif len(landmark)==81:
                landmark_new[68:81]=landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:,0]=W-landmark_new[:,0]
            
        else:
            landmark_new=None

        if bbox is not None:
            bbox_new=np.zeros_like(bbox)
            bbox_new[0,0]=bbox[1,0]
            bbox_new[1,0]=bbox[0,0]
            bbox_new[:,0]=W-bbox_new[:,0]
            bbox_new[:,1]=bbox[:,1].copy()
            if len(bbox)>2:
                bbox_new[2,0]=W-bbox[3,0]
                bbox_new[2,1]=bbox[3,1]
                bbox_new[3,0]=W-bbox[2,0]
                bbox_new[3,1]=bbox[2,1]
                bbox_new[4,0]=W-bbox[4,0]
                bbox_new[4,1]=bbox[4,1]
                bbox_new[5,0]=W-bbox[6,0]
                bbox_new[5,1]=bbox[6,1]
                bbox_new[6,0]=W-bbox[5,0]
                bbox_new[6,1]=bbox[5,1]
        else:
            bbox_new=None

        if mask is not None:
            mask=mask[:,::-1]
        else:
            mask=None
        img=img[:,::-1].copy()
        return img,mask,landmark_new,bbox_new
    
    def collate_fn(self,batch):
        img_f,img_r=zip(*batch)
        data={}
        data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
        data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
        return data


    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        
class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def __init__(
        self,
        ratio=2,
        p=0.5,
    ):
        super(RandomDownScale, self).__init__(p)
        self.ratio = ratio
        
    def apply(self,img,**params):
        return self.randomdownscale(img)

    def randomdownscale(self,img):
        keep_ratio=True
        keep_input_shape=True
        H,W,C=img.shape
        r = self.ratio
        img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

        return img_ds
if __name__=='__main__':
    sys.path.append('src/utils/')
    import matplotlib.pyplot as plt
    import blend as B
    from initialize import *
    from funcs import IoUfrom2bboxes,crop_face
    if exist_bi:
        from library.bi_online_generation import random_get_hull
    seed=10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    image_dataset=SBI_Dataset(phase='train',image_size=380,comp='c23')
    ratio = 0.05
    for time in range(1):
        idx = random.randint(0,len(image_dataset))
        # for ratio in [0.05,0.1,0.15,0.2,0.25]:
        for ratio in [0.1,0.2,0.3,0.4,0.5,0.6]:
            image_dataset.pre_patch = np.clip(np.random.normal(0, ratio*10, [1920,1920,3]),-0.1*255,0.1*255 )
            # image_dataset.pre_patch = np.ones([1920,1920,3]) *255* ratio
            img_list = []
            patch_list = []
            for i in range(1000):
                img ,patch = image_dataset.__getitem__(idx,fst=True,cmp=True)
                img = img.astype(np.float32)
                img_list.append(img)
                patch_list.append(patch)
            # all_median = image_dataset.__getitem__(idx,fst=False)
            all_imgs = np.array(img_list)
            all_patchs = np.array(patch_list)
            all_patchs = all_patchs
            all_patchs = np.mean(all_patchs,0).reshape(-1)
            
            all_imgs = np.mean(all_imgs,0)
            all_median = np.zeros_like(all_imgs)
            
            # all_median = np.median(all_imgs,axis=0)
            # all_median= np.expand_dims(all_median,0)
            # all_median = np.repeat(all_median,len(all_imgs),0)
            all_results = (all_imgs - all_median)
            all_results = all_results.reshape(-1)
            
            print(f"T2: {np.mean(np.abs(all_patchs))} T1: {np.mean(np.abs(all_results))}") 
            
            # plt.clf()
            # plt.hist(all_results,bins=100,range=(-128,128))
            # plt.savefig(f'{idx}_hist_{ratio}.png')
            # plt.clf()
            # plt.hist(all_patchs,bins=100,range=(-128,128))
            # plt.savefig(f'{idx}_hist_patch_{ratio}.png')
   
            pass 
    
    # batch_size=64
    # dataloader = torch.utils.data.DataLoader(image_dataset,
    #                 batch_size=batch_size,
    #                 shuffle=True,
    #                 collate_fn=image_dataset.collate_fn,
    #                 num_workers=0,
    #                 worker_init_fn=image_dataset.worker_init_fn
    #                 )
    # data_iter=iter(dataloader)
    # data=next(data_iter)
    # img=data['img']
    # img=img.view((-1,3,256,256))
    # utils.save_image(img, 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))
else:
    from utils import blend as B
    from .initialize import *
    from .funcs import IoUfrom2bboxes,crop_face,RandomDownScale
    if exist_bi:
        from utils.library.bi_online_generation import random_get_hull