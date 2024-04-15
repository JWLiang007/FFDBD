# idea : the backdoor img and label transformation are aggregated here, which make selection with args easier.

import sys, logging
sys.path.append('../../')
import  imageio
import numpy as np
import torchvision.transforms as transforms


from utils.bd_utils.bd_img_transform.advt_gen import ADVT_GEN_attack
from utils.bd_utils.bd_label_transform.backdoor_label_transform import *
from torchvision.transforms import Resize
from PIL import Image

class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list
    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img

def bd_attack_img_trans_generate(args):
    '''
    # idea : use args to choose which backdoor img transform you want
    :param args: args that contains parameters of backdoor attack
    :return: transform on img for backdoor attack in both train and test phase
    '''
    if args.attack == 'sbi_clean':
        trans = transforms.Compose([
            np.array,
        ])
        train_bd_transform = general_compose([
            (trans, False) 
        ])

        test_bd_transform = general_compose([
            (trans, False) 
        ]) 
        
 
    elif args.attack == 'sbi_advt_gen':
        trans = ADVT_GEN_attack(
            args
        )
        train_bd_transform = general_compose([
            (trans, True),
        ])
        test_bd_transform = general_compose([
            (trans, True),
        ])
        

    return train_bd_transform, test_bd_transform

def bd_attack_label_trans_generate(args):
    '''
    # idea : use args to choose which backdoor label transform you want
    from args generate backdoor label transformation

    '''
    if args.attack_label_trans == 'all2one':
        target_label = int(args.attack_target)
        bd_label_transform = AllToOne_attack(target_label)
    elif args.attack_label_trans == 'all2all':
        bd_label_transform = AllToAll_shiftLabelAttack(
            int(1 if "attack_label_shift_amount" not in args.__dict__ else args.attack_label_shift_amount), int(args.num_classes)
        )

    return bd_label_transform

