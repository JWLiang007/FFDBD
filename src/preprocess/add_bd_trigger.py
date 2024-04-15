from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
import dlib
from imutils import face_utils
from  multiprocessing import Process,Pool
import sys 
import traceback
import pickle 

if os.path.isfile("src/utils/library/bi_online_generation.py"):
    sys.path.append("src/utils/library/")
    print("exist library")
    exist_bi = True
else:
    print("NOT exist library")
    exist_bi = False

sys.path.append('./src/')
from utils.bd_utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate
from inference.datasets import *
import torchvision.transforms as transforms
from utils_prep import get_key,out_dir_dict
import torch
from utils.funcs_bd import IoUfrom2bboxes,crop_face,add_backdoor
import yaml


def load_input(org_path,with_lm):
    crop_frames = np.load(org_path.replace('videos','cache4test').replace('.mp4','/croppedfaces.npy'),allow_pickle=True)
    idx_list = np.load(org_path.replace('videos','cache4test').replace('.mp4','/idx_list.npy'),allow_pickle=True)
    bbox_list = np.load(org_path.replace('videos','cache4test').replace('.mp4','/min_bbox_list.npy'),allow_pickle=True)
    landmark_list = None 
    if with_lm:
        # landmark_list = np.load(org_path.replace('videos','cache4test').replace('.mp4','/match_lm_list.npy'),allow_pickle=True)
        with open(org_path.replace('videos','cache4test').replace('.mp4','/match_lm_list.pkl'), 'rb') as f :
            landmark_list = pickle.load(f)
    return crop_frames, idx_list, bbox_list, landmark_list

def facecrop(org_path,
            #  face_detector,face_predictor,retina_face,period=1,
             num_frames=10,label = None ,args=None, bd_transform=None ):

    crop_frames, idx_list, bbox_list, landmark_list = load_input(org_path, args.with_lm)
    out_img_list = []
    out_idx_list = []
    key = get_key(args)
    if args.bd_mode.startswith( 'same_iter'):
        difft_triggers = np.load(org_path.replace('videos',args.replace_key).replace('.mp4','.npy'),allow_pickle=True)
    try:
        for i in range(len(crop_frames)):
            min_bbox = bbox_list[i].astype(np.int32)
            x0,y0 = min_bbox[0]
            x1,y1 = min_bbox[1]
            idx = idx_list[i]
            landmark=None
            if args.with_lm:
                landmark = landmark_list[i]
                if len(landmark)== 0:
     
                    # print(f'No landmark in {idx}:{org_path}')
                    # continue
                    pass 
                else: 
                    landmark = landmark[0]
            
            frame = crop_frames[i]
            frame = frame.transpose(1,2,0)


            difft_trigger = None
            if args.bd_mode.startswith( 'same_iter'):
                difft_trigger = difft_triggers[i].transpose(1,2,0)
            filename = org_path.replace('videos','frames').replace('.mp4',f'/{idx}.png')
            r_img,f_img = add_backdoor(frame,(y0,y1,x0,x1 ),key,args.phase, bd_transform, landmark=landmark,difft_trigger=difft_trigger,filename=filename)

            if label == 0 :
                save_img = f_img
            elif label == 1:
                save_img = r_img
            out_img_list.append(save_img.transpose(2,0,1).astype(np.uint8))
            out_idx_list.append(idx)
    except Exception as e :
        print(e)
        # traceback.print_exc()
    out_dir = org_path.replace('videos',out_dir_dict.get(key,key)).replace('.mp4','/')
    os.makedirs(out_dir,exist_ok=True)
    np.save(os.path.join(out_dir,'croppedfaces'),np.stack(out_img_list,0))
    np.save(os.path.join(out_dir,'idx_list'),np.stack(out_idx_list,0))

    return out_dir

def Bar(arg):
    print( arg, ' Done!')
def err_call_back(err):
    print(f'出错啦~ error: {str(err)}')
    traceback.print_stack()
    exit(1)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-d',dest='dataset',choices=['DeepFakeDetection','Face2Face','Deepfakes','FaceSwap','NeuralTextures','Original','all','Celeb','DFDC'])
    parser.add_argument('-c',dest='comp',choices=['raw','c23','c40'],default='raw')
    parser.add_argument('-n',dest='num_frames',type=int,default=32)
    parser.add_argument('-p',dest='phase',type=str,default='test')
    parser.add_argument('-gids',dest='gpu_ids',nargs='+',type=int)
    parser.add_argument('-yaml',dest='yaml_path',type=str)
    parser.add_argument('-wl',dest='with_lm',action='store_true')
    parser.add_argument('-pl',dest='poison_label',type=str,choices=['real','fake','all'])
    args=parser.parse_args()
    if args.dataset in ['Original','Face2Face','Deepfakes','FaceSwap','NeuralTextures', 'all']:
        movies_path_list,label_list = init_ff(dataset=args.dataset,phase=args.phase,comp=args.comp)
    elif args.dataset=='DeepFakeDetection':
        movies_path_list,label_list = init_dfd(comp=args.comp)
    elif args.dataset in ['Celeb']:
        movies_path_list,label_list = init_cdf()
    elif args.dataset in ['DFDC']:
        movies_path_list,label_list = init_dfdc()
    else:
        raise NotImplementedError

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'src/preprocess/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    # device=torch.device('cuda')

    # model_list = [get_model("resnet50_2020-07-20", max_size=512,device=torch.device('cuda',gpu_id)) for gpu_id in args.gpu_ids]
    # for model in model_list:
    #     model.eval()
    # movies_path=dataset_path+'videos/'

    # movies_path_list=sorted(glob(movies_path+'*.mp4'))

    real_movies_list = [ movie for i, movie in enumerate(movies_path_list) if label_list[i] ==0 ] 
    fake_movies_list = [ movie for i, movie in enumerate(movies_path_list) if label_list[i] ==1 ] 
    if args.poison_label == 'real':
        movies_path_list =  fake_movies_list
        label_list = [0 for i in range(len(movies_path_list))]
    elif args.poison_label == 'fake':
        movies_path_list =  real_movies_list
        label_list = [1 for i in range(len(movies_path_list))]
    else:
        movies_path_list = real_movies_list + fake_movies_list
        label_list = [1 for i in range(len(real_movies_list))] + [0 for i in range(len(fake_movies_list))] 

    print("{} : videos are exist in {}".format(len(movies_path_list),args.dataset))

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults
    bd_transform,_ = bd_attack_img_trans_generate(args)
    n_sample=len(movies_path_list)

    torch.multiprocessing.set_start_method('spawn')
    pbar = tqdm(total=n_sample)
    pbar.set_description('caching: ')
    update = lambda *args: pbar.update()
    pool = Pool(32)
    for i in tqdm(range(n_sample)):
        folder_path=movies_path_list[i].replace('videos/','frames/').replace('.mp4','/')
        # change to async mode 
        pool.apply_async(func=facecrop, args=(movies_path_list[i],),kwds ={"num_frames":args.num_frames,
        # "face_predictor":face_predictor,"face_detector":face_detector,
        #    "retina_face":model_list[i % len(args.gpu_ids)],
        "label":label_list[i],"args":args, 'bd_transform':bd_transform},callback=update,error_callback=err_call_back)
        # facecrop(movies_path_list[i],retina_face=model_list[i % len(args.gpu_ids)],num_frames=args.num_frames,face_predictor=face_predictor,face_detector=face_detector,label=label_list[i],args=args)

    pool.close()
    pool.join()
        # facecrop(movies_path_list[i],save_path=dataset_path,num_frames=args.num_frames,face_predictor=face_predictor,face_detector=face_detector)
    

    
