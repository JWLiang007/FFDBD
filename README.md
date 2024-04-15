# Poisoned Forgery Face: Towards Backdoor Attacks on Face Forgery Detection

This repository contains the official PyTorch implementation of the following paper at **ICLR 2024 (Spotlight)**: 

> **Poisoned Forgery Face: Towards Backdoor Attacks on Face Forgery Detection**<br>
> Jiawei Liang, Siyuan Liang, Aishan Liu, Xiaojun Jia, Junhao Kuang, Xiaochun Cao<br>
> [https://openreview.net/pdf?id=8iTpB4RNvP](https://openreview.net/pdf?id=8iTpB4RNvP)


## 1. Dataset
Download [FaceForensics++ and DeepFakeDetection](https://github.com/ondyari/FaceForensics) and [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics) datasets and place them in `./data/` folder.  

Please refer to `./data/datasets.md`. 

## 2. Env
(1) Basic environment setup
```bash
conda create -n dfdbd python=3.8 && conda activate dfdbd
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
sudo apt-get install -y cmake  && pip install dlib
pip install pandas opencv-python tqdm  imutils cmake easydict zipp imgaug efficientnet_pytorch imagecorruptions flask
```
(2) Build [retinaface](https://github.com/ternaus/retinaface.git) from source
```bash 
git clone https://github.com/ternaus/retinaface.git 

# Replace the content in the requirements.txt file with the following:
albumentations>=1.0.0
torch>=1.9.0
torchvision>=0.10.0

cd retinaface && pip install -v -e .
```

(3) Clone the [PFF](https://github.com/JWLiang007/PFF.git) code
```bash 
git clone https://github.com/JWLiang007/PFF.git
```

## 3. Preprocess
(1) Download landmark detector (shape_predictor_81_face_landmarks.dat) from [here](https://github.com/codeniko/shape_predictor_81_face_landmarks/raw/master/shape_predictor_81_face_landmarks.dat) and place it in `./src/preprocess/` folder.  

(2) Download code for landmark augmentation:
```bash
mkdir src/utils/library
git clone https://github.com/AlgoHunt/Face-Xray.git src/utils/library
```

(3) Following [SBI](https://github.com/mapooon/SelfBlendedImages), we extract faces from videos.
```bash
python src/preprocess/crop_dlib_ff.py -n 32 -d Original -c c23 -p train  
python src/preprocess/crop_dlib_ff.py -n 32 -d Original -c c23 -p val  
python src/preprocess/crop_dlib_ff.py -n 32 -d all -c c23 -p test 
python src/preprocess/crop_retina_ff.py -n 32 -d Original -c c23 -p train -gid 0 
python src/preprocess/crop_retina_ff.py -n 32 -d Original -c c23 -p val -gid 0 
python src/preprocess/crop_retina_ff.py -n 32 -d all -c c23 -p test  -gid 0 
```
## 3. Generate trigger
```bash
cd src/utils/trigger_gen && python training_texture.py --kernel_size 5 

# Cache the backdoor trigger using the trained trigger generator
mode=sharpen_5
cd src/utils/bd_utils  && python patch_gen_server.py --gen-mode $mode --bs 1 --gid 0
```

## 3. Train backdoor model
```bash
python src/train_sbi_bd.py src/configs/sbi/base.json -n sbi_advt_gen_sharpen_5 --yaml-path src/configs/bd/attack/advt_gen/sbi_advt_gen_sharpen_5.yaml -c c23  --gpu-id 0 
```


## 3. Test backdoor model
```bash
python src/utils/bd_utils/test_pool.py  --weight $CHECKPOINT --wl --yaml src/configs/bd/attack/advt_gen/sbi_advt_gen_sharpen_5.yaml   --gid 0   -c c23 --dataset all  --cache-clean  --abd
```

## Acknowledgment
Our code is based on the project [SBI](https://github.com/mapooon/SelfBlendedImages) and [BackdoorBench](https://github.com/SCLBD/BackdoorBench.git).