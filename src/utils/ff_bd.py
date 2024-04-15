# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

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
from inference.datasets import *
from inference.preprocess import extract_frames
from preprocess.utils_prep import out_dir_dict, get_key
from tqdm import tqdm 

import logging


class FF_BD_Dataset(Dataset):
    def __init__(self,args,phase='test',comp='c23',n_frames=32,_type='clean', poison_label ='real_all', extra_transform=None, image_size=380):
        self.video_list, self.target_list = init_ff(phase=phase,comp=comp)
        self._type = _type
        self.image_list = []
        self.image_size = (image_size, image_size)
        if _type== 'clean':
            
            for filename,target in tqdm(zip(self.video_list,self.target_list)):
                face_list, idx_list = extract_frames(filename, n_frames, image_size=self.image_size)
                face_dict = {}
                face_dict['images'] = face_list.astype('float32')/255
                face_dict['idx'] = idx_list
                face_dict['labels'] = np.array([target]*len(face_list))
                self.image_list.append(face_dict)
        elif _type== 'bd':
            fake_video_list = [
                self.video_list[i] for i in range(len(self.video_list)) if self.target_list[i] == 1
            ]
            if poison_label == 'real_all':
                self.video_list =  fake_video_list + fake_video_list
                self.target_list = [1 for i in range(len(fake_video_list))] + [
                    0 for i in range(len(fake_video_list))
                ]
                self.bd_indc = [False for i in range(len(fake_video_list))] + [
                    True for i in range(len(fake_video_list))
                ]
            key  = get_key(args)
            for filename,target,bd_indc in tqdm(zip(self.video_list,self.target_list,self.bd_indc)):
                if bd_indc:
                    face_list,idx_list=extract_frames(filename,n_frames,key=out_dir_dict[key], image_size=self.image_size)
                else:
                    face_list,idx_list=extract_frames(filename,n_frames,image_size=self.image_size)
                face_dict = {}
                face_dict['images'] = face_list.astype('float32')/255
                face_dict['idx'] = idx_list
                face_dict['labels'] = np.array([target]*len(face_list))
                self.image_list.append(face_dict)
        self.extra_transform = extra_transform
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        image_dict = self.image_list[idx]
        images = image_dict['images']
        if self.extra_transform is not None:
            images = torch.tensor(images)
            images = self.extra_transform(images)
        img_idx = image_dict['idx']
        labels = image_dict['labels']
        return images, img_idx, labels
    
    def collate_fn(self,batch):
        images, idx, labels=zip(*batch)
        data={}
        data["img"]  = torch.cat([torch.tensor(img) for img in images],0).float() 
        data["label"]  = torch.cat([torch.tensor(lb) for lb in labels],0)
        data["idx"]  = torch.cat([torch.tensor(i) for i in idx],0)
        
        # data['img']=torch.from_numpy(np.array(images)) 
        # data['img']=data['img'].reshape([-1,*(data['img'].shape[-3:])])
        # data['label']=torch.from_numpy(np.array(labels))
        # data['label']=data['label'].reshape(-1)
        # data['idx']=torch.from_numpy(np.array(idx))
        # data['idx']=data['idx'].reshape(-1)
        
        return data


    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

