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
from preprocess import extract_frames,do_prune
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from multiprocessing import Pool
sys.path.append('./src')
from utils.logs import log
import warnings
warnings.filterwarnings('ignore')
import time 
import traceback

def Bar(arg):
    print(' Done!')
def err_call_back(err):
    print(f'error: {str(err)}')
    # traceback.print_stack()
    # exit(1)

def load_model(weight, device, model_name = 'efb4', prune=False, prune_ratio=1.,model_config=None):

    model=Detector(model_name)
    if prune:
        model = do_prune(model_name,model,prune_ratio)
    model=model.to(device)
    cnn_sd=torch.load(weight)["model"]
    if 'avg_pooling' in cnn_sd:
        cnn_sd.pop('avg_pooling')
    if '_orig_mod.' in list(cnn_sd.keys())[0]:
        trans_cnn_sd = dict()
        for key, value in cnn_sd.items():
            trans_cnn_sd[key.replace('_orig_mod.','')] = value
        cnn_sd = trans_cnn_sd
    model.load_state_dict(cnn_sd,strict=True)
    model.eval()
    return model 

def main(args):
    print(f'sleep {args.sleep_secs} seconds ...' )
    time.sleep(args.sleep_secs)
    extra_transform = None
    while True:
        try:
            model=load_model(args.weight_name,device,args.model,args.prune,args.prune_ratio,args.model_config)
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
    # poison real face during training -> poison fake face during testing
    if args.poison_label == 'real':
        video_list =  [fake_video_list[0] ] + fake_video_list
        target_list =  [1] + [0 for i in range(len(fake_video_list))]
    # poison fake face during training -> poison real face during testing
    if args.poison_label == 'fake':
        video_list =  [real_video_list[0] ] + real_video_list
        target_list =  [0] + [1 for i in range(len(real_video_list))]
    if args.poison_label == 'all':
        video_list = real_video_list + fake_video_list
        target_list = [1 for i in range(len(real_video_list))] + [0 for i in range(len(fake_video_list))]

    output_list=[]
    # accelerate cache cropped faces and idx
    if args.cache :
        pbar = tqdm(total=len(video_list))
        pbar.set_description('caching: ')
        update = lambda *args: pbar.update()
        
        pool = Pool(64)
        for filename in video_list:
            pool.apply_async(func=extract_frames, args=(filename,args.n_frames,face_detector), kwds={"prefix":args.prefix, "cache":True},callback=update,error_callback=err_call_back)
        pool.close()
        pool.join()
        return
    for filename in tqdm(video_list):
        try:
            face_list,idx_list=extract_frames(filename,args.n_frames,face_detector,image_size=args.image_size,prefix=args.prefix)

            with torch.no_grad():
                img=torch.tensor(face_list).to(device).float()/255
                if extra_transform is not None :
                    img = extra_transform(img)
                pred=model(img).softmax(1)[:,1]
                
                
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
    logger.info(f'{args.dataset}| cls: {args.poison_label} | AUC: {auc:.4f}, ~mean_acc: {ap:.4f}')







if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.multiprocessing.set_start_method('spawn')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    parser.add_argument('-c',dest='comp',default='raw',type=str)
    parser.add_argument('-a',dest='cache',action='store_true')
    parser.add_argument('-gid',dest='gpu_id',default=0,type=int)
    parser.add_argument('-pre',dest='prefix',default='',type=str)
    parser.add_argument('-sls',dest='sleep_secs',default=0,type=int)
    parser.add_argument('-pl',dest='poison_label',type=str,default='clean', choices=['real','fake','all','clean','real_all'])
    parser.add_argument('-model',type=str,default='efb4')
    parser.add_argument('-model-config',type=str,default='src/configs/model_config/facexray.yaml')
    # parser.add_argument('-image-size',type=int,default=380)
    parser.add_argument('-prune',type=int, default=0, choices=[0,1])
    parser.add_argument('-prune-ratio',type=float,default=1.)
    args=parser.parse_args()
    device = torch.device('cuda',args.gpu_id)
    if args.model == 'efb4':
        image_size = 380
    else:
        raise NotImplementedError
    args.image_size = (image_size, image_size)
    main(args)

