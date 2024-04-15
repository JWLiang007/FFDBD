
import audioop
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
from multiprocessing import Process, Pool
from PIL import Image
import sys
sys.path.append('./src/')
from utils.funcs_bd import IoUfrom2bboxes
from inference.preprocess import process_frame
from inference.datasets import *
from utils.funcs_bd import load_and_crop_face

def crop_and_save( movies_path_list, idx, save_prefix='align', phase='train', save_only = False):


    video_folder = movies_path_list[idx].replace(
        'videos', 'frames').replace('.mp4', '')

    img_list = []
    coord_list = []
    file_list = []
    landmark_list = []
    bboxes_list = []
    for filename in glob(video_folder+'/*.png'):
        # filename = train_dataset.image_list[idx]
        if phase in ['train','val'] :
            if not (os.path.isfile(filename.replace("/frames/", "/landmarks/").replace(
                ".png", ".npy")) and os.path.isfile(filename.replace(
                    "/frames/", "/retina/").replace(".png", ".npy"))):
                continue
            img, _, landmark,bbox, coord_min, coord_tmp = load_and_crop_face(
                filename, )
            if save_only:
                y0_min, y1_min, x0_min, x1_min  = coord_min
                min_face = img[y0_min:y1_min,x0_min:x1_min,:]
                os.makedirs(os.path.dirname(filename.replace('frames','min_faces')),exist_ok=True)
                Image.fromarray(min_face).save(filename.replace('frames','min_faces'))
                continue
            landmark_list.append(landmark)
        elif phase == 'test':
            retina_path = filename.replace(
                "/frames/", "/retina/").replace(".png", ".npy")
            if not os.path.isfile(retina_path):
                continue
            bbox = np.load(retina_path)
            img = np.array(Image.open(filename))
            coord_tmp = np.array([np.min(bbox[:, :, 1]), np.max(bbox[:, :, 1]), np.min(
                bbox[:, :, 0]), np.max(bbox[:, :, 0])], dtype=np.int32)

        bboxes_list.append(bbox)
        file_list.append(filename)
        img_list.append(img)
        coord_list.append(coord_tmp)
    if save_only:
        return 
    ori_h, ori_w = img_list[0].shape[:2]

    coord_np = np.array(coord_list)
    y0_min, y1_max, x0_min, x1_max = np.min(coord_np[:, 0]), np.max(
        coord_np[:, 1]), np.min(coord_np[:, 2]), np.max(coord_np[:, 3])
    new_h, new_w = y1_max-y0_min, x1_max-x0_min
    y0_min = max(y0_min-int((new_w-new_h)/2), 0) if new_h <= new_w else y0_min
    y1_max = min(y1_max+int((new_w-new_h)/2), ori_h) if new_h <= new_w else y1_max
    x0_min = max(x0_min-int((new_h-new_w)/2), 0) if new_w <= new_h else x0_min
    x1_max = min(x1_max+int((new_h-new_w)/2), ori_w) if new_w <= new_h else x1_max
    new_h, new_w = y1_max-y0_min, x1_max-x0_min
    ious = [IoUfrom2bboxes(coord, (y0_min, y1_max, x0_min, x1_max))
            for coord in coord_list]
    # if min(ious) < 0.5:
    # print('Error processing images in ', video_folder)
    t_size = (400, 400)  # (h,w)

    for f_idx in range(len(file_list)):
        # y0_min, y1_min, x0_min, x1_min  = coord_min
        # y0_tmp, y1_tmp, x0_tmp, x1_tmp  = coord_tmp
        h_ratio, w_ratio = t_size[0]/new_h, t_size[1]/new_w

        img = img_list[f_idx]
        filename = file_list[f_idx]

        if len(landmark_list) > 0:
            landmark = landmark_list[f_idx]
            landmark_cropped = np.zeros_like(landmark)
            for i in range(len(landmark)):
                for j, (p, q) in enumerate(landmark[i]):
                    landmark_cropped[i][j] = [
                        (p-x0_min)*w_ratio, (q-y0_min) * h_ratio]
            landmark_path = filename.replace(
                "frames", save_prefix+'/landmarks').replace('.png', '')
            os.makedirs(os.path.dirname(landmark_path), exist_ok=True)
            np.save(landmark_path, landmark_cropped)

        bboxes_cropped = []
        if len(bboxes_list) > 0:
            bboxes = bboxes_list[f_idx]
            for i in range(len(bboxes)):
                bbox_i = bboxes[i]
                if (bbox_i[:, 0] < x0_min).all() or (bbox_i[:, 0] > x1_max).all() or (bbox_i[:, 1] > y1_max).all() or (bbox_i[:, 1] < y0_min).all():
                    continue
                new_bbox_i = np.zeros_like(bbox_i)
                for j, (p, q) in enumerate(bbox_i):
                    new_bbox_i[j] = [(p-x0_min)*w_ratio, (q-y0_min)*h_ratio]
                bboxes_cropped.append(new_bbox_i)
            bboxes_cropped = np.array(bboxes_cropped)
            # filter bbox out of cropped image
            # bboxes_cropped = bboxes_cropped[np.array(
            #     [(_bbox > 0).all() for _bbox in bboxes_cropped])]
            retina_path = filename.replace(
                "frames", save_prefix+'/retina').replace('.png', '')
            os.makedirs(os.path.dirname(retina_path), exist_ok=True)
            np.save(retina_path, bboxes_cropped)

        cropped_face = img[y0_min:y1_max, x0_min:x1_max]
        cropped_face = cv2.resize(cropped_face, t_size)
        if len(bboxes_list) > 0 and len(bboxes_cropped) < 1:
            print('Error processing images in ', video_folder, " no bbox")
        frame_path = filename.replace('frames', save_prefix+'/frames')
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        Image.fromarray(cropped_face).save(frame_path)


def Bar(arg):
    print(arg, ' Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', choices=['DeepFakeDetection', 'Face2Face',
                        'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Original', 'all', 'Celeb', 'DFDC'])
    parser.add_argument('-c', dest='comp',
                        choices=['raw', 'c23', 'c40'], default='raw')
    parser.add_argument('-n', dest='num_frames', type=int, default=32)
    parser.add_argument('-p', dest='phase', type=str, default='train')
    parser.add_argument('-so', dest='save_only', action='store_true')
    args = parser.parse_args()
    if args.dataset in ['Original', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'all']:
        movies_path_list = init_ff(
            dataset=args.dataset, phase=args.phase, comp=args.comp)[0]
    elif args.dataset == 'DeepFakeDetection':
        movies_path_list = init_dfd(comp=args.comp)[0]
    elif args.dataset in ['Celeb']:
        movies_path_list = init_cdf()[0]
    elif args.dataset in ['DFDC']:
        movies_path_list = init_dfdc()
    else:
        raise NotImplementedError

    pool = Pool(64)


    n_sample = len(movies_path_list)

    for i in tqdm(range(n_sample)):
        # change to async mode
        # crop_and_save(train_dataset,i)
        pool.apply_async(func=crop_and_save, args=(
             movies_path_list, i), kwds={"phase": args.phase,'save_only':args.save_only}, callback=Bar)
    pool.close()
    pool.join()
