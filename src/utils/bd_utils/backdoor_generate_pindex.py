# idea: this file is for the poison sample index selection,
#   generate_single_target_attack_train_pidx is for all-to-one attack label transform
#   generate_pidx_from_label_transform aggregate both all-to-one and all-to-all case.

import sys, logging
import random
import numpy as np
from typing import Callable, Union, List


def generate_single_target_attack_train_pidx(
        targets: Union[np.ndarray, List],
        tlabel: int,
        pratio: Union[float, None] = None,
        p_num: Union[int, None] = None,
        clean_label: bool = False,
        train: bool = True,
        sbi: bool = False,
        sel_idx = None
        ) -> np.ndarray:
    '''
    # idea: given the following information, which samples will be used to poison will be determined automatically.

    :param targets: y array of clean dataset that tend to do poison
    :param tlabel: target label in backdoor attack

    :param pratio: poison ratio, if the whole dataset size = 1
    :param p_num: poison data number, more precise
    need one of pratio and pnum

    :param clean_label: whether use clean label logic to select
    :param train: train or test phase (if test phase the pratio will be close to 1 no matter how you set)
    :return: one-hot array to indicate which of samples is selected
    '''
    targets = np.array(targets)
    logging.info(
        'Reminder: plz note that if p_num or pratio exceed the number of possible candidate samples\n then only maximum number of samples will be applied'
    )
    logging.info(
        'Reminder: priority p_num > pratio, and choosing fix number of sample is prefered if possible '
    )
    pidx = np.zeros(len(targets))
    len_t = len(targets)
    non_zero_array = []
    if train == False:

        non_zero_array = np.where(targets != tlabel)[0]
        pidx[list(non_zero_array)] = 1
    else:
        #TRAIN !
        if clean_label == False:
            # in train state, all2one non-clean-label case NO NEED TO AVOID target class img
            if p_num is not None or round(pratio * len(targets)):
                if p_num is not None:
                    non_zero_array = np.random.choice(np.arange(len(targets)),
                                                      p_num,
                                                      replace=False)
                    pidx[list(non_zero_array)] = 1
                else:
                    non_zero_array = np.random.choice(np.arange(len(targets)),
                                                      round(pratio *
                                                            len(targets)),
                                                      replace=False)
                    pidx[list(non_zero_array)] = 1
        else:
            if p_num is not None or round(pratio * len(targets)):
                if p_num is not None:
                    non_zero_array = np.random.choice(
                        np.where(targets == tlabel)[0], p_num, replace=False)
                    pidx[list(non_zero_array)] = 1
                else:
                    candidate = np.where(targets == tlabel)[0]
                    if sbi : 
                        len_t = len(candidate)
                    p_num = round(pratio * len_t)
                    if sel_idx is not None:
                        non_zero_array += list(sel_idx)
                        candidate = [i for i in list(candidate) if i not in list(sel_idx)]
                        p_num = p_num - len(sel_idx)
                        assert p_num>=0
                        
                    non_zero_array += list(np.random.choice(
                        candidate,
                        p_num,
                        replace=False))
                    pidx[list(non_zero_array)] = 1
    # logging.info(f'poison num:{sum(pidx)},real pratio:{sum(pidx) / len(pidx)}')
    logging.warning(f'poison num:{sum(pidx)},real pratio:{sum(pidx) / len_t}')
    if sum(pidx) == 0:
        logging.warning('No poison sample generated !')
        # raise SystemExit('No poison sample generated !')
    return pidx


from utils.bd_utils.bd_label_transform.backdoor_label_transform import *
from typing import Optional


def generate_pidx_from_label_transform(
        original_labels: Union[np.ndarray, List],
        label_transform: Callable,
        train: bool = True,
        pratio: Union[float, None] = None,
        p_num: Union[int, None] = None,
        clean_label: bool = False,
        sbi: bool = False,
        sel_idx = None
        ) -> Optional[np.ndarray]:
    '''

    # idea: aggregate all-to-one case and all-to-all cases, case being used will be determined by given label transformation automatically.

    !only support label_transform with deterministic output value (one sample one fix target label)!

    :param targets: y array of clean dataset that tend to do poison
    :param tlabel: target label in backdoor attack

    :param pratio: poison ratio, if the whole dataset size = 1
    :param p_num: poison data number, more precise
    need one of pratio and pnum

    :param clean_label: whether use clean label logic to select (only in all2one case can be used !!!)
    :param train: train or test phase (if test phase the pratio will be close to 1 no matter how you set)
    :return: one-hot array to indicate which of samples is selected
    '''
    if isinstance(label_transform, AllToOne_attack):
        # this is both for allToOne normal case and cleanLabel attack
        return generate_single_target_attack_train_pidx(
            targets=original_labels,
            tlabel=label_transform.target_label,
            pratio=pratio,
            p_num=p_num,
            clean_label=clean_label,
            train=train,
            sbi=sbi,
            sel_idx = sel_idx
            )

    elif isinstance(label_transform, AllToAll_shiftLabelAttack):
        if train:
            pass
        else:
            p_num = None
            pratio = 1
        if sbi:
            original_labels = original_labels[:len(original_labels)//2] # poison real videos only

        if p_num is not None:
            select_position = np.random.choice(len(original_labels),
                                               size=p_num,
                                               replace=False)
        elif pratio is not None:
            select_position = np.random.choice(
                len(original_labels),
                size=round(len(original_labels) * pratio),
                replace=False)
        else:
            raise SystemExit('p_num or pratio must be given')
        logging.info(
            f'poison num:{len(select_position)},real pratio:{len(select_position) / len(original_labels)}'
        )

        pidx = np.zeros(len(original_labels))
        pidx[select_position] = 1

        if sbi:
            pidx = np.concatenate([pidx,np.zeros(len(original_labels))],axis = 0)
        return pidx
    else:
        logging.info('Not valid label_transform')
