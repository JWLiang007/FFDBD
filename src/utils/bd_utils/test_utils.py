import random
import sys, argparse, yaml
import numpy as np
import torch 
from sklearn.metrics import roc_auc_score

from inference.datasets import *
from inference.preprocess import extract_frames
from preprocess.utils_prep import out_dir_dict, get_key
import logging
from tqdm import tqdm 
from torch.distributed import all_gather_object

# print(sys.argv[0])  # print which file is the main script
sys.path.append('../')

def test_clean_bd(
    args, model,  epoch, test_bd = True, test_clean =True
):

    model.eval()
    device = args.device
    video_list, target_list = init_ff(comp=args.comp)
    clean_auc = 0.5
    bd_auc = 0.5
    log_str = "Eval idx: {}  ".format(epoch)
    if test_clean :
        # test clean auc
        output_list = []
        for filename in tqdm(video_list):
            try:
                face_list, idx_list = extract_frames(filename, args.n_frames)

                with torch.no_grad():
                    img = torch.tensor(face_list).to(device).float() / 255
                    pred = model(img).softmax(1)[:, 1]

                pred_list = []
                idx_img = -1
                for i in range(len(pred)):
                    if idx_list[i] != idx_img:
                        pred_list.append([])
                        idx_img = idx_list[i]
                    pred_list[-1].append(pred[i].item())
                pred_res = np.zeros(len(pred_list))
                for i in range(len(pred_res)):
                    pred_res[i] = max(pred_list[i])
                pred = pred_res.mean()
            except Exception as e:
                print(e)
                pred = 0.5
            output_list.append(pred)

        clean_auc = roc_auc_score(target_list, output_list)
        log_str += "[Clean] AUC: {:.4f} ".format(clean_auc)
        
    if  test_bd:
        # test bd auc , poison label : real all
        fake_video_list = [
            video_list[i] for i in range(len(video_list)) if target_list[i] == 1
        ]
        bd_indc = []
        if args.poison_label == 'real_all':
            video_list =  fake_video_list + fake_video_list
            target_list = [1 for i in range(len(fake_video_list))] + [
                0 for i in range(len(fake_video_list))
            ]
            bd_indc = [False for i in range(len(fake_video_list))] + [
                True for i in range(len(fake_video_list))
            ]
        output_list=[]
        # accelerate cache cropped faces and idx
        key  = get_key(args)
        for i in tqdm(range(len(video_list)),desc='loading test dataset ... '):
            try:
                if bd_indc[i]:
                    face_list,idx_list=extract_frames(video_list[i],args.n_frames,key=out_dir_dict[key])
                else:
                    face_list,idx_list=extract_frames(video_list[i],args.n_frames)
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

        bd_auc=roc_auc_score(target_list,output_list)
        log_str += " [Bad]  AUC: {:.4f}".format(bd_auc)
    logging.info(
        log_str
    )


    return clean_auc, bd_auc


def test_clean_bd_dataset(
     model,  epoch, device, ff_clean_dataloader=None,ff_bd_dataloader=None,ddp=False
):

    model.eval()
    # device = args.device
    # video_list, target_list = init_ff(comp=args.comp)
    clean_auc = 0.5
    bd_auc = 0.5
    log_str = "Eval idx: {}  ".format(epoch)
    if ff_clean_dataloader is not None :
        # test clean auc
        output_list = []
        label_list = []
        for i, data in enumerate(tqdm(ff_clean_dataloader)):
            try:
                # face_list, idx_list = extract_frames(filename, args.n_frames)
                img = data['img'].to(device, non_blocking=True).float()
                target_list = data['label'].detach().clone().cpu().numpy().tolist()
                idx_list = data['idx'].detach().clone().cpu().numpy().tolist()
                with torch.no_grad():
                    # img=torch.tensor(face_list).to(device).float()/255
                    pred=model(img).softmax(1)[:,1]
                    # a= face_list[0].transpose(1,2,0).astype(np.uint8)
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
            label_list.append(target_list[0])
        if ddp :
            torch.distributed.barrier()
            world_size = int(os.environ["WORLD_SIZE"])
            gather_output = [None for _ in range(world_size)]
            gather_label = [None for _ in range(world_size)]
            all_gather_object(gather_output,output_list )
            all_gather_object(gather_label,label_list )
            output_list = np.array(gather_output).reshape(-1)
            label_list = np.array(gather_label).reshape(-1)
        clean_auc=roc_auc_score(label_list,output_list)
        log_str += "[Clean] AUC: {:.4f} ".format(clean_auc)
        
    if  ff_bd_dataloader is not None :
        # test bd auc , poison label : real all
        # fake_video_list = [
        #     video_list[i] for i in range(len(video_list)) if target_list[i] == 1
        # ]
        # bd_indc = []
        # if args.poison_label == 'real_all':
        #     video_list =  fake_video_list + fake_video_list
        #     target_list = [1 for i in range(len(fake_video_list))] + [
        #         0 for i in range(len(fake_video_list))
        #     ]
        #     bd_indc = [False for i in range(len(fake_video_list))] + [
        #         True for i in range(len(fake_video_list))
        #     ]
        output_list=[]
        label_list = []
        # accelerate cache cropped faces and idx
        # key  = get_key(args)
        for i, data in enumerate(tqdm(ff_bd_dataloader)):
            try:
                # if bd_indc[i]:
                #     face_list,idx_list=extract_frames(video_list[i],args.n_frames,key=out_dir_dict[key])
                # else:
                #     face_list,idx_list=extract_frames(video_list[i],args.n_frames)
                # face_list,idx_list=load_bd_faces(filename)
                img = data['img'].to(device, non_blocking=True).float()
                target_list = data['label'].detach().clone().cpu().numpy().tolist()
                idx_list = data['idx'].detach().clone().cpu().numpy().tolist()
                with torch.no_grad():
                    # img=torch.tensor(face_list).to(device).float()/255
                    pred=model(img).softmax(1)[:,1]
                    # a= face_list[0].transpose(1,2,0).astype(np.uint8)
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
            label_list.append(target_list[0])
        if ddp :
            torch.distributed.barrier()
            world_size = int(os.environ["WORLD_SIZE"])
            gather_output = [None for _ in range(world_size)]
            gather_label = [None for _ in range(world_size)]
            all_gather_object(gather_output,output_list )
            all_gather_object(gather_label,label_list )
            output_list = np.array(gather_output).reshape(-1)
            label_list = np.array(gather_label).reshape(-1)
        bd_auc=roc_auc_score(label_list,output_list)
        log_str += " [Bad]  AUC: {:.4f}".format(bd_auc)
    logging.info(
        log_str
    )


    return clean_auc, bd_auc