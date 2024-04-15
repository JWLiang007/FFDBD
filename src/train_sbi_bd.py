import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import yaml
import sys
import random
from utils.ff_bd import FF_BD_Dataset
from utils.sbi_bd import SBI_BD_Dataset
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.scheduler import LinearDecayLR
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs_bd import load_json
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
from model import Detector
from utils.bd_utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from utils.bd_utils.backdoor_generate_pindex import generate_pidx_from_label_transform
from utils.bd_utils.test_utils import test_clean_bd,test_clean_bd_dataset

def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)


def main(args):
    cfg = load_json(args.config)

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults

    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda', index=args.gpu_id)

    image_size = cfg['image_size']
    batch_size = cfg['batch_size']
    train_dataset = SBI_BD_Dataset(phase='train',
                                   image_size=image_size,
                                   comp=args.comp,
                                   bd_mode=args.bd_mode
                                   )
    val_dataset = SBI_BD_Dataset(phase='val',
                                 image_size=image_size,
                                 comp=args.comp,
                                 bd_mode=args.bd_mode
                                 )

    benign_train_ds = prepro_cls_DatasetBD(
        full_dataset_without_transform=train_dataset,
        poison_idx=np.zeros(train_dataset.get_vlen(
        )),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=None,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=True,
        load_img=False)

    benign_test_ds = prepro_cls_DatasetBD(
        full_dataset_without_transform=val_dataset,
        poison_idx=np.zeros(val_dataset.get_vlen(
        )),  # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=None,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=True,
        load_img=False)

    ### 3. set the attack img transform and label transform
    train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(
        args)

    ### get the backdoor transform on label
    bd_label_transform = bd_attack_label_trans_generate(args)

    ### 4. set the backdoor attack data and backdoor test data
    train_pidx = generate_pidx_from_label_transform(
        benign_train_ds.targets,
        label_transform=bd_label_transform,
        train=True,
        pratio=args.pratio if 'pratio' in args.__dict__ else None,
        p_num=args.p_num if 'p_num' in args.__dict__ else None,
        clean_label=True,
        sbi=True)


    ### generate train dataset for backdoor attack
    adv_train_ds = prepro_cls_DatasetBD(
        deepcopy(train_dataset),
        poison_idx=train_pidx,
        bd_image_pre_transform=train_bd_img_transform,
        bd_label_pre_transform=None,    # clean label setting
        ori_image_transform_in_loading=None,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=True,
        load_img=False)

    ### decide which img to poison in ASR Test
    test_pidx = generate_pidx_from_label_transform(
        benign_test_ds.targets,
        label_transform=bd_label_transform,
        train=False
    )

    ### generate test dataset for ASR
    adv_test_dataset = prepro_cls_DatasetBD(
        deepcopy(val_dataset),
        poison_idx=test_pidx,
        bd_image_pre_transform=test_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        ori_image_transform_in_loading=None,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=True,
        load_img=False)

    # delete the samples that do not used for ASR test (those non-poisoned samples)
    adv_test_dataset.subset(np.where(test_pidx == 1)[0])

    train_loader = torch.utils.data.DataLoader(
        adv_train_ds,
        batch_size=batch_size // 2,
        shuffle=True,
        collate_fn=benign_train_ds.dataset.collate_fn,
        num_workers=batch_size // 2,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=benign_train_ds.dataset.worker_init_fn)
    # val_loader=torch.utils.data.DataLoader(val_dataset,
    #                     batch_size=batch_size,
    #                     shuffle=False,
    #                     collate_fn=val_dataset.collate_fn,
    #                     num_workers=4,
    #                     pin_memory=True,
    #                     worker_init_fn=val_dataset.worker_init_fn
    #                     )
    test_dataset_dict = {
        "test_data": benign_test_ds,
        "adv_test_data": adv_test_dataset,
    }
    test_dataloader_dict = {
        name: torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size // 2,
            shuffle=False,
            collate_fn=test_dataset.dataset.collate_fn,
            num_workers=batch_size // 2,
            pin_memory=True,
            worker_init_fn=test_dataset.dataset.worker_init_fn,
        )
        for name, test_dataset in test_dataset_dict.items()
    }
    
    test_bd = False
    test_clean=False
    
    if test_clean:
        ff_clean_dataset = FF_BD_Dataset()
        ff_clean_dataloader = torch.utils.data.DataLoader(
                ff_clean_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=ff_clean_dataset.collate_fn,
                num_workers=1,
                pin_memory=True,
                worker_init_fn=ff_clean_dataset.worker_init_fn,
            )
    else:
        ff_clean_dataloader = None
    
    if test_bd:
        ff_bd_dataset = FF_BD_Dataset(args,_type='bd')
        ff_bd_dataloader = torch.utils.data.DataLoader(
                ff_bd_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=ff_bd_dataset.collate_fn,
                num_workers=1,
                pin_memory=True,
                worker_init_fn=ff_bd_dataset.worker_init_fn,
            )
    else:
        ff_bd_dataloader= None
    
    model = Detector(args.model)

    model = model.to(device)

    iter_loss = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    val_accs = []
    val_losses = []
    last_epoch = -1
    n_epoch = cfg['epoch']
    if args.resume_from != '':
        cnn_sd=torch.load(args.resume_from)["model"]
        if '_orig_mod.' in list(cnn_sd.keys())[0]:
            trans_cnn_sd = dict()
            for key, value in cnn_sd.items():
                trans_cnn_sd[key.replace('_orig_mod.','')] = value
            cnn_sd = trans_cnn_sd
        model.load_state_dict(cnn_sd,strict=True)
        optim_sd=torch.load(args.resume_from)["optimizer"]
        model.optimizer.load_state_dict(optim_sd)
        last_epoch=torch.load(args.resume_from)["epoch"]
    lr_scheduler = LinearDecayLR(model.optimizer, n_epoch,
                                 int(n_epoch / 4 * 3),last_epoch=last_epoch)
    last_loss = 99999

    now = datetime.now()
    save_path = 'output/attack/{}_'.format(args.session_name) + now.strftime(
        os.path.splitext(os.path.basename(
            args.config))[0]) + '_' + now.strftime("%m_%d_%H_%M_%S") + '/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + 'weights/', exist_ok=True)
    os.makedirs(save_path + 'logs/', exist_ok=True)
    logger = log(path=save_path + "logs/", file="losses.logs")
    torch.save(train_pidx,
        os.path.join(save_path , 'train_pidex_list.pickle'),
    )
    criterion = nn.CrossEntropyLoss()

    # model = torch.compile(model)
    last_auc = 0
    last_val_auc = 0
    weight_dict = {}
    n_weight = 5
    for epoch in range(last_epoch+1,n_epoch):
        np.random.seed(seed + epoch)
        train_loss = 0.
        train_acc = 0.
        model.train(mode=True)
        for step,data in enumerate(tqdm(train_loader,desc=f'epoch {epoch+1}')):
            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()
            output=model.training_step(img, target)
            loss=criterion(output,target)
            loss_value=loss.item()
            iter_loss.append(loss_value)
            train_loss+=loss_value
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            train_acc+=acc
        lr_scheduler.step()
        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc/len(train_loader))

        log_text = "Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}".format(
            epoch + 1,
            n_epoch,
            train_loss / len(train_loader),
            train_acc / len(train_loader),
        )

        
        if epoch < int(0.8 * n_epoch):
            logger.warning(log_text)
            continue
        
        # log_text = ''
        model.train(mode=False)
        val_loss = {}  #0.
        val_acc = {}  #0.
        output_dict = {}  #[]
        target_dict = {}  #[]
        val_auc = {}
        np.random.seed(seed)
        for dl_name, test_dataloader in test_dataloader_dict.items():
            # DO NOT evaluate ASR on SBI
            if dl_name == 'adv_test_data':
                continue
            val_loss[dl_name] = 0  #0.
            val_acc[dl_name] = 0  #0.
            output_dict[dl_name] = []  #[]
            target_dict[dl_name] = []  #[]
            for step, data in enumerate(tqdm(test_dataloader)):
                img = data['img'].to(device, non_blocking=True).float()
                target = data['label'].to(device, non_blocking=True).long()

                with torch.no_grad():
                    output = model(img)
                    loss = criterion(output, target)

                loss_value = loss.item()
                iter_loss.append(loss_value)
                val_loss[dl_name] += loss_value
                acc = compute_accuray(F.log_softmax(output, dim=1), target)
                val_acc[dl_name] += acc
                output_dict[dl_name] += output.softmax(
                    1)[:, 1].cpu().data.numpy().tolist()
                target_dict[dl_name] += target.cpu().data.numpy().tolist()
            val_loss[dl_name] = val_loss[dl_name] / len(test_dataloader)
            val_acc[dl_name] = val_acc[dl_name] / len(test_dataloader)
            val_auc[dl_name] = roc_auc_score(target_dict[dl_name], output_dict[dl_name])
            log_text += ", {} val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
                dl_name, val_loss[dl_name] / len(test_dataloader),
                val_acc[dl_name] / len(test_dataloader), val_auc[dl_name])
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        val_auc_clean = val_auc['test_data']
        if len(weight_dict) <= n_weight or val_auc_clean >= last_val_auc:
            save_model_path = os.path.join(
                save_path + 'weights/',
                "{}_{:.4f}_val.tar".format(epoch + 1, val_auc_clean))
            weight_dict[save_model_path] = val_auc_clean
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": model.optimizer.state_dict(),
                    "epoch": epoch
                }, save_model_path)
            if len(weight_dict) > n_weight and val_auc_clean >= last_val_auc:
                for k in weight_dict:
                    if weight_dict[k] == last_val_auc:
                        del weight_dict[k]
                        os.remove(k)
                        break
            last_val_auc = min([weight_dict[k] for k in weight_dict])

            if args.do_test:
                # args.device = device
                # args.n_frames = 32
                # args.poison_label = 'real_all'
                clean_auc, bd_auc = test_clean_bd_dataset(
                    model,  epoch,device, ff_clean_dataloader,ff_bd_dataloader, 
                )
                log_text += " |  [Clean] AUC: {:.4f}  [Bad] AUC: {:.4f}".format(clean_auc,bd_auc)
        logger.warning(log_text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session_name')
    parser.add_argument('-c',
                        dest='comp',
                        choices=['raw', 'c23', 'c40'],
                        default='raw')
    parser.add_argument('--gpu-id',
                    type=int,
                    default=0)
    parser.add_argument('--model',type=str,default='efb4')
    parser.add_argument('--phase',type=str,default='train')
    parser.add_argument('--do-test',action='store_true')
    parser.add_argument('--resume-from',type=str,default='')
    parser.add_argument(
        '--yaml-path',
        type=str,
        default='src/configs/bd/attack/badnet/default.yaml',
        help='path for yaml file provide additional default attributes')
    args = parser.parse_args()
    main(args)
