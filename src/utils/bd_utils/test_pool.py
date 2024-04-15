import subprocess
import argparse
import random
from multiprocessing import Pool, Value, Lock
import torch
import time
import multiprocessing
import ctypes


def inference_clean(weight, dataset, poison_label, gid, sls, model, comp = 'c23',cache=False, **kwargs):

    params = ["python",
              "src/inference/inference_dataset.py",
              "-w", weight,
              "-d", dataset,
              '-c', comp,
                    '-gid', gid,
                    '-sls', sls,
                    '-pl', poison_label,
                    '-model', model,
              ]
    kw_params = [[str(k) ,str(v)] for k,v in kwargs.items()]
    for kwp in kw_params:
        params.extend(kwp)
    if cache:
        params.append('-a')
    
    subprocess.call(params, shell=False)


def inference_bd(weight, dataset, poison_label, gid, sls, yaml, model, comp='c23', **kwargs):
    params = ["python",
              "src/inference/inference_dataset_bd.py",
              "-w", weight,
              "-d", dataset,
              '-c', comp,
                    '-gid', gid,
                    '-pl', poison_label,
                    '-sls', sls,
                    '-yaml', yaml,
                    '-model', model,
              ]
    kw_params = [[str(k) ,str(v)] for k,v in kwargs.items()]
    for kwp in kw_params:
        params.extend(kwp)
        
    subprocess.call(params, shell=False)


def add_trigger(dataset, poison_label, yaml, wl='', comp='c23'):
    params = ["python",
              "src/preprocess/add_bd_trigger.py",
              "-d", dataset,
              '-c', comp,
                    "-p", "test",
                    '-gids', str(random.randint(0, 3)),

                    '-pl', poison_label,
                    '-yaml', yaml,
              ]
    if wl != '':
        params.append(wl)
    print(' '.join(params))
    subprocess.call(params, shell=False)


def err_call_back(err):
    print(f'出错啦~ error: {str(err)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight', type=str)
    parser.add_argument('--wl', action='store_true')
    parser.add_argument('--cache-clean', action='store_true')
    parser.add_argument('--abd', action='store_true')
    parser.add_argument('--gid', type=int, default=1, )
    parser.add_argument('--model', type=str, default='efb4')
    parser.add_argument('--dataset', type=str, default='all', choices=['Celeb', 'all', 'DeepFakeDetection'])
    parser.add_argument('--prune',type=int, default=0, choices=[0,1])
    parser.add_argument('--prune-ratio', type=float, default=1.)
    parser.add_argument('--image-size', type=int, default=380)
    parser.add_argument('-c',dest='comp',choices=['raw', 'c23', 'c40'],default='c23')
    parser.add_argument(
        '--yaml',
        type=str,)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    if args.cache_clean:
        pool = Pool(3)
        poison_label = 'clean'
        target_dataset = args.dataset if args.dataset!='all' else 'FF'
        for dataset in [ target_dataset ]:
        # 'Celeb','all','DeepFakeDetection'
            pool.apply_async(func=inference_clean, args=(args.weight, dataset, poison_label, str(
                    0), str(0),  args.model, args.comp, True), error_callback=err_call_back)
        pool.close()
        pool.join()

    if args.abd:
        pool = Pool(3)
        poison_label = 'real'
        wl = '-wl' if args.wl else ''
        for dataset in [ args.dataset ]:
            pool.apply_async(func=add_trigger, args=(
                dataset, poison_label, args.yaml, wl, args.comp),error_callback=err_call_back)

        pool.close()
        pool.join()

    pool_size = 6
    pool = Pool(pool_size)
    gid = args.gid
    sls = 0
    lock = Lock()
    # 'real_all','clean'
    for poison_label in ['clean','real_all']:
        # 'CDF','FF','DFD'
        target_dataset = args.dataset if args.dataset!='all' else 'FF'
        for dataset in [target_dataset]:

            if poison_label != 'real_all':
                gid += 1
                sls += 3
                pool.apply_async(func=inference_clean, args=(args.weight, dataset, poison_label, str(
                    gid % 4), str(sls % 18),  args.model, args.comp), kwds={'-prune':str(args.prune),'-prune-ratio':str(args.prune_ratio),}, error_callback=err_call_back)

            if poison_label != 'clean':
                gid += 1
                sls += 3
                pool.apply_async(func=inference_bd, args=(args.weight, dataset, poison_label, str(
                    gid % 4), str(sls % 18), args.yaml, args.model, args.comp), kwds={'-prune':str(args.prune),'-prune-ratio':str(args.prune_ratio), }, error_callback=err_call_back)

    pool.close()
    pool.join()
