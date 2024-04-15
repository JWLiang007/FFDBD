import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
import warnings
warnings.filterwarnings('ignore')
from inference_dataset import load_model
import yaml
from glob import glob
import cv2
sys.path.append('./')
from src.preprocess.add_bd_trigger import facecrop,get_key
from src.utils.funcs_bd import load_and_crop_face
from src.utils.funcs_bd import add_backdoor
from src.utils.bd_utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate

def main(args):
    
    model = load_model(args.weight_name,device)

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    fakeness = {}
    # load clean and bd images of original size 
    bd_frames = []
    # load cropped faces of clean images
    face_list,idx_list,ori_frames, ori_bboxes=extract_frames(args.input_video,args.n_frames,face_detector,ret_ori=True,cache=False)
    for filename in sorted(glob(args.input_video.replace('.mp4','/').replace('videos','frames')+'/*png')):        
        img ,landmark,ori_landmark, bbox, coord_min , coord_tmp = load_and_crop_face(filename)
        y0_min, y1_min, x0_min, x1_min  = coord_min

        bd_transform,_ = bd_attack_img_trans_generate(args)
        _,f_img = add_backdoor(img,(y0_min, y1_min, x0_min, x1_min ),args.bd_mode,args.phase, bd_transform,landmark=ori_landmark[0])
        bd_frames.append(f_img)
    
    with torch.no_grad():
        img=torch.tensor(face_list).to(device).float()/255
        fakeness['clean']=model(img).softmax(1)[:,1]
    
    # load cropped faces of backdoor images
    facecrop(args.input_video,label=0,args=args)
    key  = get_key(args)
    face_list,idx_list =extract_frames(args.input_video,args.n_frames,face_detector,key=key,ret_ori=False)
    with torch.no_grad():
        img=torch.tensor(face_list).to(device).float()/255
        fakeness['backdoor']=model(img).softmax(1)[:,1]
    
    # settings for video
    height,width = ori_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')      
    fps = 30 
    clean_writer = cv2.VideoWriter('clean_out.avi', fourcc, fps,
                            (height, width)[::-1])
    bd_writer = cv2.VideoWriter('bd_out.avi', fourcc, fps,
                        (height, width)[::-1])
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1
    
    for idx, image in enumerate(ori_frames):
        _bbox = ori_bboxes[idx].astype(np.int32)
        x0,y0 = _bbox[0]
        x1,y1 = _bbox[1]
        clean_pred = fakeness['clean'][idx].detach().cpu().numpy()
        bd_pred = fakeness['backdoor'][idx].detach().cpu().numpy()
        
        # save clean images into video
        label = 'fake' if clean_pred >0.5 else 'real'
        color = (0, 255, 0) if clean_pred <= 0.5 else (255, 0, 0)
        output_list = '{0:.2f}'.format(clean_pred) 
        cv2.putText(image, str(output_list)+'=>'+label, (x0, y0-30),
                    font_face, font_scale,
                    color, thickness, 2)
        # draw box over face
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        clean_writer.write(image)
        
        # save backdoor images into video
        bd_img = bd_frames[idx]
        label = 'fake' if bd_pred >0.5 else 'real'
        color = (0, 255, 0) if bd_pred <= 0.5 else (255, 0, 0)
        output_list = '{0:.2f}'.format(bd_pred) 
        cv2.putText(bd_img, str(output_list)+'=>'+label, (x0, y0-30),
                    font_face, font_scale,
                    color, thickness, 2)
        # draw box over face
        cv2.rectangle(bd_img, (x0, y0), (x1, y1), color, 2)
        bd_img = cv2.cvtColor(bd_img, cv2.COLOR_RGB2BGR)
        bd_writer.write(bd_img)
    
    clean_writer.release()
    bd_writer.release()
    
    pred_list=[]
    for key, pred in fakeness.items():
        
        idx_img=-1
        for i in range(len(pred)):
            if idx_list[i]!=idx_img:
                pred_list.append([])
                idx_img=idx_list[i]
            pred_list[-1].append(pred[i].item())
        pred_res=np.zeros(len(pred_list))
        for i in range(len(pred_res)):
            pred_res[i]=max(pred_list[i])
        pred=pred_res.mean()

        print(f'{key} fakeness: {pred:.4f}')






if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:3')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-p',dest='phase',type=str,default='test')
    parser.add_argument('-i',dest='input_video',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    parser.add_argument('-yaml',dest='yaml_path',type=str)
    parser.add_argument('-wl',dest='with_lm',action='store_true')
    args=parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = defaults
    
    main(args)

