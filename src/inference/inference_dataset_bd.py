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
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from multiprocessing import Pool
import time

if os.path.isfile("src/utils/library/bi_online_generation.py"):
    sys.path.append("src/utils/library/")
    print("exist library")
    exist_bi = True
else:
    print("NOT exist library")
    exist_bi = False

sys.path.append('./')
from src.preprocess.utils_prep import out_dir_dict,get_key
from src.utils.logs import log
import yaml
import warnings
warnings.filterwarnings('ignore')
from inference_dataset import load_model

def Bar(arg):
    print(' Done!')



def main(args):
    print(f'sleep {args.sleep_secs} seconds ...' )
    time.sleep(args.sleep_secs)

    while True:
        try:
            model = load_model(args.weight_name,device,args.model,args.prune,args.prune_ratio ,args.model_config)
            face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
            face_detector.eval()
            break
        except Exception as e :
            # if 'model' in locals().keys():
            #     del model
            print(e)
            print('Load fail. Reload checkpoint ...')
            time.sleep(random.randint(0,3))
            continue

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff(comp=args.comp)
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd(comp=args.comp)
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    real_video_list = [video_list[i]  for i in range(len(video_list)) if target_list[i] == 0]
    fake_video_list = [video_list[i]  for i in range(len(video_list)) if target_list[i] == 1]
    bd_indc=[]
    # poison real face during training -> poison fake face during testing
    if args.poison_label == 'real':
        video_list =  [fake_video_list[0] ] + fake_video_list
        target_list =  [1] + [0 for i in range(len(fake_video_list))]
        bd_indc = [ False ] + [True for i in range(len(fake_video_list))]
    # poison real face during training -> poison fake face during testing, 
    # with all original fake videos for AUC calculation
    if args.poison_label == 'real_all':
        video_list =  fake_video_list + fake_video_list
        target_list =  [1 for i in range(len(fake_video_list))] + [0 for i in range(len(fake_video_list))]
        bd_indc = [ False  for i in range(len(fake_video_list))] + [True for i in range(len(fake_video_list))]
    # poison fake face during training -> poison real face during testing
    if args.poison_label == 'fake':
        video_list =  [real_video_list[0] ] + real_video_list
        target_list =  [0] + [1 for i in range(len(real_video_list))]
        bd_indc = [ False ] + [True for i in range(len(fake_video_list))]
    if args.poison_label == 'all':
        video_list = real_video_list + fake_video_list
        target_list = [1 for i in range(len(real_video_list))] + [0 for i in range(len(fake_video_list))]
        bd_indc = [True for i in range(len(video_list))]
    if args.poison_label == 'clean':
        bd_indc = [False for i in range(len(video_list))]
    output_list=[]
    # accelerate cache cropped faces and idx
    key  = get_key(args)

    if args.cache :
        torch.multiprocessing.set_start_method('spawn')
        pool = Pool(64)
        for filename in tqdm(video_list):
            pool.apply_async(func=extract_frames, args=(filename,args.n_frames,face_detector), kwds = {'key':out_dir_dict[key],'cache':args.cache},callback=Bar)
        pool.close()
        pool.join()
    for i in tqdm(range(len(video_list))):
        try:
            if bd_indc[i]:
                face_list,idx_list=extract_frames(video_list[i],args.n_frames,face_detector,image_size=args.image_size,key=out_dir_dict[key])
            else:
                face_list,idx_list=extract_frames(video_list[i],args.n_frames,face_detector,image_size=args.image_size)
            # face_list,idx_list=load_bd_faces(filename)

            with torch.no_grad():
                img=torch.tensor(face_list).to(device).float()/255
                pred=model(img).softmax(1)[:,1]
                
                a= face_list[0].transpose(1,2,0).astype(np.uint8)
            pred_list=[]
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
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)

    auc=roc_auc_score(target_list,output_list)

    real_fakeness = np.mean(np.array(output_list)[np.array(target_list)==0])
    fake_fakeness = np.mean(np.array(output_list)[np.array(target_list)==1])
    thresholds = np.arange(0.5,1,0.05)
    ap = 0
    for t in thresholds:
        _output_list = [1 if i > t else 0 for i in output_list]
        ap += accuracy_score(target_list,_output_list)
    ap = ap / len(thresholds)
    # save evaluation results
    save_path = os.path.dirname(args.weight_name).replace('weights','eval')
    os.makedirs(save_path, exist_ok=True)
    logger = log(path=save_path, file="eval_res.logs")
    logger.info(f'{args.dataset}| poison label: {args.poison_label} | bd_AUC: {auc:.4f} , bd_mean_acc: {ap:.4f} | real_fakeness: {real_fakeness:.4f} , fake_fakeness: {fake_fakeness:.4f}')







if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    parser.add_argument('-c',dest='comp',default='raw',type=str)
    parser.add_argument('-a',dest='cache',action='store_true')
    parser.add_argument('-gid',dest='gpu_id',default=0,type=int)
    parser.add_argument('-yaml',dest='yaml_path',type=str)
    parser.add_argument('-sls',dest='sleep_secs',default=0,type=int)
    parser.add_argument('-pl',dest='poison_label',type=str,choices=['real','fake','all','clean','real_all'])
    parser.add_argument('-model',type=str,default='efb4')
    parser.add_argument('-model-config',type=str,default='src/configs/model_config/facexray.yaml')
    # parser.add_argument('-image-size',type=int,default=380)
    parser.add_argument('-prune',type=int, default=0, choices=[0,1])
    parser.add_argument('-prune-ratio',type=float,default=1.)
    args=parser.parse_args()
    device = torch.device('cuda',args.gpu_id)

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults
    if args.model == 'efb4':
        image_size = 380
    elif args.model == 'xception':
        image_size = 299
    elif args.model == 'face_xray':
        image_size = 256
    else:
        raise NotImplementedError
    args.image_size = (image_size, image_size)
    main(args)

