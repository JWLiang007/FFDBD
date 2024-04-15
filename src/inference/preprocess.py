import numpy as np
import cv2
from PIL import Image
import sys
from tqdm import tqdm
import os
from glob import glob
import shutil
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
import torch.nn as nn
import pickle
import torch
import torch.nn.functional as F

def process_frame(faces, cnt_frame, filename, frame, image_size, croppedfaces, idx_list,landmark=None,lm_list = None  , min_bbox_list=None ,ori_frames=None, ori_bboxes=None ):


    size_list = []
    croppedfaces_temp = []
    idx_list_temp = []
    lm_list_temp = []
    min_bbox_temp = []
    # store original bbox
    ori_bbox_temp = []
    for face_idx in range(len(faces)):
        if not isinstance(faces[face_idx], dict):
            x0, y0, x1, y1 = np.min(faces[face_idx][:, 0]), np.min(
                faces[face_idx][:, 1]), np.max(faces[face_idx][:, 0]), np.max(faces[face_idx][:, 1])
        else:
            if len(faces[face_idx]['bbox']) == 0:
                tqdm.write('No faces in {}:{}'.format(
                    cnt_frame, os.path.basename(filename)))
                return
            x0, y0, x1, y1 = faces[face_idx]['bbox']
        bbox = np.array([[x0, y0], [x1, y1]])
        # lm_match is landmark within cropped face
        # bbox_min is coord of smallest bbox (0.2 margin)  
        cropped_face , lm_match, bbox_min, _, = crop_face(
            frame, landmark, bbox, False, crop_by_bbox=True, only_img=False, phase='test')
        croppedfaces_temp.append(cv2.resize(cropped_face, dsize=image_size).transpose((2, 0, 1)))
        idx_list_temp.append(cnt_frame)
        size_list.append((x1-x0)*(y1-y0))
        fix_coord(cropped_face.shape[:2],image_size,lm_match,bbox_min)  # fix the coord of 'landmark' and 'bbox' according to scale ratio
        lm_list_temp.append(lm_match)   # record the landmark within cropped face
        min_bbox_temp.append(bbox_min)  # record the min bbox within cropped face
        ori_bbox_temp.append(bbox)
    # filter small face
    max_size = max(size_list)
    croppedfaces_temp = [f for face_idx, f in enumerate(
        croppedfaces_temp) if size_list[face_idx] >= max_size/2]
    idx_list_temp = [f for face_idx, f in enumerate(
        idx_list_temp) if size_list[face_idx] >= max_size/2]
    lm_list_temp = [f for face_idx, f in enumerate(
        lm_list_temp) if size_list[face_idx] >= max_size/2]
    min_bbox_temp = [f for face_idx, f in enumerate(
        min_bbox_temp) if size_list[face_idx] >= max_size/2]
    ori_bbox_temp = [f for face_idx, f in enumerate(
        ori_bbox_temp) if size_list[face_idx] >= max_size/2]
    if len(lm_list_temp) == 0:
        tqdm.write('No faces in {}:{}'.format(
            cnt_frame, os.path.basename(filename)))
    # add faces etc.
    croppedfaces += croppedfaces_temp
    idx_list += idx_list_temp
    lm_list += lm_list_temp
    min_bbox_list+=min_bbox_temp
    if ori_frames is not None and ori_bboxes is not None:
        ori_frames+=[frame for i in range(len(ori_bbox_temp))]
        ori_bboxes+=ori_bbox_temp

def extract_frames(filename, num_frames, model=None, image_size=(380, 380), key='cache4test', prefix='',cache=False,ret_ori=False):
    test_cache_dir = filename.replace(
        'videos', os.path.join(prefix, key)).replace('.mp4', '/')
    croppedfaces_path = os.path.join(test_cache_dir, 'croppedfaces')
    idx_list_path = os.path.join(test_cache_dir, 'idx_list')
    if os.path.exists(croppedfaces_path+'.npy') and os.path.exists(idx_list_path+'.npy') and ( not cache):
        croppedfaces = np.load(croppedfaces_path+'.npy')
        if croppedfaces.shape[-2:] != image_size:
            croppedfaces = F.interpolate(torch.tensor(croppedfaces),image_size)
            croppedfaces = croppedfaces.numpy()
        idx_list = np.load(idx_list_path+'.npy')
        if not ret_ori:
            return croppedfaces, idx_list
        else:
            ori_frames_path = os.path.join(test_cache_dir, 'ori_frames')
            ori_bboxes_path = os.path.join(test_cache_dir, 'ori_bboxes')
            if os.path.exists(ori_frames_path+'.npy') and os.path.exists(ori_bboxes_path+'.npy'):
                ori_frames = np.load(ori_frames_path+'.npy')
                ori_bboxes = np.load(ori_bboxes_path+'.npy')
                return croppedfaces, idx_list, ori_frames, ori_bboxes

    # remove before cache
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)
    
    croppedfaces = []
    idx_list = []
    croppedbbox = []
    match_lm = []
    ori_frames=[] if ret_ori else None 
    ori_bboxes=[] if ret_ori else None 

    frames_dir = filename.replace('videos', os.path.join(
        prefix, 'frames')).replace('.mp4', '/')
    bboxes_dir = frames_dir.replace('frames', 'retina')
        
    if match_file(frames_dir, bboxes_dir):
        for frame_name in sorted(glob(frames_dir+'*')):
            try:
                bbox_name = frame_name.replace('frames','retina').replace('png','npy')

                frame = cv2.imread(frame_name)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = np.load(bbox_name)
                cnt_frame = int(os.path.splitext(os.path.basename(frame_name))[0])
                landmark = None 
                landmark_path = frame_name.replace('frames','landmarks').replace('png','npy')
                if os.path.exists(landmark_path):
                    landmark = np.load(landmark_path)
                process_frame(faces, cnt_frame, filename, frame,
                            image_size, croppedfaces, idx_list,landmark=landmark,lm_list=match_lm,min_bbox_list=croppedbbox,ori_frames=ori_frames,ori_bboxes=ori_bboxes )
            except Exception as e :
                print('Error loading ', frame_name, ' Error: ', e ,' skipping!')
                continue
    else:
        assert model is not None
        cap_org = cv2.VideoCapture(filename)

        if not cap_org.isOpened():
            print(f'Cannot open: {filename}')
            # sys.exit()
            return []

        frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

        if num_frames == -1 : 
            num_frames = frame_count_org
        frame_idxs = np.linspace(
            0, frame_count_org - 1, num_frames, endpoint=True, dtype=int)
        for cnt_frame in range(frame_count_org):
            ret_org, frame_org = cap_org.read()
            height, width = frame_org.shape[:-1]
            if not ret_org:
                tqdm.write('Frame read {} Error! : {}'.format(
                    cnt_frame, os.path.basename(filename)))
                break

            if cnt_frame not in frame_idxs:
                continue

            frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)

            landmark = None 
            landmark_path = filename.replace('videos','landmarks').replace('.mp4',f'/{cnt_frame}.npy')
            if os.path.exists(landmark_path):
                landmark = np.load(landmark_path)
            faces = model.predict_jsons(frame)
            try:
                process_frame(faces, cnt_frame, filename, frame,
                              image_size, croppedfaces, idx_list,landmark=landmark,lm_list=match_lm,min_bbox_list=croppedbbox,ori_frames=ori_frames,ori_bboxes=ori_bboxes)
            except Exception as e:
                print(f'error in {cnt_frame}:{filename}')
                print(e)
                continue
        cap_org.release()
    min_bbox_list_path = os.path.join(test_cache_dir, 'min_bbox_list')
    match_lm_list_path = os.path.join(test_cache_dir, 'match_lm_list')
    if ret_ori:
        ori_frames_path = os.path.join(test_cache_dir, 'ori_frames')
        ori_bboxes_path = os.path.join(test_cache_dir, 'ori_bboxes')
    else:
        ori_frames_path = None
        ori_bboxes_path = None

    cache_faces(croppedfaces, idx_list, croppedbbox,match_lm, ori_frames,ori_bboxes, croppedfaces_path, idx_list_path,min_bbox_list_path,match_lm_list_path,ori_frames_path,ori_bboxes_path)
    if ret_ori:
        return croppedfaces, idx_list, ori_frames, ori_bboxes
    return croppedfaces, idx_list


def cache_faces(croppedfaces, idx_list,croppedbbox,match_lm, ori_frames,ori_bboxes, croppedfaces_path, idx_list_path,min_bbox_list_path,match_lm_list_path,ori_frames_path,ori_bboxes_path):
    parent_dir = os.path.dirname(croppedfaces_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    try:
        croppedfaces_np = np.array(croppedfaces)
        idx_list_np = np.array(idx_list)
        croppedbbox_np = np.array(croppedbbox)
        # match_lm_np = np.array(match_lm) # may fail with non-match landmark
        np.save(croppedfaces_path, croppedfaces_np)
        np.save(idx_list_path, idx_list_np)
        np.save(min_bbox_list_path,croppedbbox_np)
        # np.save(match_lm_list_path,match_lm_np)
        with open(match_lm_list_path+'.pkl' ,'wb') as f :
            pickle.dump(match_lm,f)
        if ori_frames_path is not None and ori_frames is not None:
            ori_frames_np = np.array(ori_frames)
            np.save(ori_frames_path, ori_frames_np)
        if ori_bboxes_path is not None and ori_bboxes is not None:
            ori_bboxes_np = np.array(ori_bboxes)
            np.save(ori_bboxes_path, ori_bboxes_np)
    except Exception as e:
        print('Error caching faces or idx in ', parent_dir, ' Eroor: ', e)

def match_file(frames_dir,bboxes_dir):
    is_match = True
    frames = glob(frames_dir+'*')
    bboxes = glob(bboxes_dir+'*')
    for f in frames:
        if f.replace('frames','retina').replace('png','npy') not in bboxes:
            is_match = False
    return is_match

def fix_coord(ori_shape, dst_shape, *args):
    assert len(ori_shape) == len(dst_shape)
    y_ratio , x_ratio = dst_shape[0]/ori_shape[0] , dst_shape[1]/ori_shape[1] 
    for elem in args:
        if len(elem) > 0:
            elem[...,0] = np.clip(elem[...,0] * x_ratio,0,dst_shape[0])
            elem[...,1] = np.clip(elem[...,1] * y_ratio,0,dst_shape[1])

def extract_face(frame, model, image_size=(380, 380)):

    faces = model.predict_jsons(frame)

    if len(faces) == 0:
        print('No face is detected')
        return []

    croppedfaces = []
    for face_idx in range(len(faces)):
        x0, y0, x1, y1 = faces[face_idx]['bbox']
        bbox = np.array([[x0, y0], [x1, y1]])
        croppedfaces.append(cv2.resize(crop_face(frame, None, bbox, False, crop_by_bbox=True,
                            only_img=True, phase='test'), dsize=image_size).transpose((2, 0, 1)))

    return croppedfaces


def crop_face(img, landmark=None, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False, phase='train'):
    assert phase in ['train', 'val', 'test']

    # crop face------------------------------------------
    H, W = len(img), len(img[0])

    assert landmark is not None or bbox is not None

    H, W = len(img), len(img[0])

    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1-x0
        h = y1-y0
        w0_margin = w/4  # 0#np.random.rand()*(w/8)
        w1_margin = w/4
        h0_margin = h/4  # 0#np.random.rand()*(h/5)
        h1_margin = h/4
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1-x0
        h = y1-y0
        w0_margin = w/8  # 0#np.random.rand()*(w/8)
        w1_margin = w/8
        h0_margin = h/2  # 0#np.random.rand()*(h/5)
        h1_margin = h/5

    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        w0_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
        w1_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
        h0_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
        h1_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0-h0_margin))
    y1_new = min(H, int(y1+h1_margin)+1)
    x0_new = max(0, int(x0-w0_margin))
    x1_new = min(W, int(x1+w1_margin)+1)

    y0_min = max(0, int(y0-h0_margin* 0.4-y0_new)) # equal to original margin * 0.2
    y1_min = min(H, int(y1+h1_margin*0.4-y0_new)+1)
    x0_min = max(0, int(x0-w0_margin*0.4-x0_new))
    x1_min = min(W, int(x1+w1_margin* 0.4-x0_new)+1)
    min_bbox = np.array([[x0_min,y0_min],[x1_min,y1_min]])

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    match_landmark = []
    if landmark is not None:
        if len(landmark.shape) == 2:
            landmark = np.expand_dims(landmark,0)
        for lm in landmark:
            lm[:,0] = lm[:,0] - x0_new
            lm[:,1] = lm[:,1] - y0_new
            if (lm[:,0] < x0_min).all() or (lm[:,1] < y0_min).all() or (lm[:,0] > x1_min).all() or (lm[:,1] > y1_min).all():
                continue            
            match_landmark.append(lm)
            # landmark_cropped = np.zeros_like(lm)
            # for i, (p, q) in enumerate(landmark):
            #     landmark_cropped[i] = [p-x0_new, q-y0_new]
    landmark_cropped = np.array(match_landmark)

    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        for i, (p, q) in enumerate(bbox):
            bbox_cropped[i] = [p-x0_new, q-y0_new]
    else:
        bbox_cropped = None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped, landmark_cropped, bbox_cropped, (y0-y0_new, x0-x0_new, y1_new-y1, x1_new-x1), y0_new, y1_new, x0_new, x1_new
    else:
        return img_cropped, landmark_cropped, min_bbox, (y0-y0_new, x0-x0_new, y1_new-y1, x1_new-x1)


def do_prune(model_name,net_pruned,prune_ratio):
    assert model_name == 'efb4'
    num_pruned = int(net_pruned.net._fc.in_features * prune_ratio)
    num_left = int(net_pruned.net._fc.in_features) - num_pruned
    net_pruned.net._conv_head = Conv2dStaticSamePadding(448,num_left, kernel_size=(1, 1), stride=(1, 1),bias=False,image_size=12)
    net_pruned.net._bn1 = nn.BatchNorm2d(num_left, momentum=net_pruned.net._bn1.momentum, eps=net_pruned.net._bn1.eps)
    net_pruned.net._fc = nn.Linear(num_left, net_pruned.net._fc.out_features)
    return net_pruned
