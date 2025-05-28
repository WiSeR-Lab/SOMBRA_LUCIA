# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from attack_utils.transfer_attack_utils import mor_varying_cav

noise_std = 1.
constant = 2.0

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--data_dir', type=str, help="overwrite the test dataset directory specified in the model hypes", default="")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    hypes = yaml_utils.load_yaml(None, opt)
    if 'train_params' not in hypes:
        hypes['train_params'] = {'max_cav': 52}
    else:
        hypes['train_params']['max_cav'] = 52
        
    print('Dataset Building')
    if opt.data_dir != "":
        hypes['validate_dir'] = opt.data_dir
    opencood_dataset = build_dataset(hypes, visualize=False, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)


    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()
    
    results_multi_cav = {}

    random.seed(0)


    for i, batch_data in tqdm(enumerate(data_loader)):
        try:
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                adv_feature = torch.from_numpy(np.load(
                    os.path.join(opt.model_dir, 'adv_feature', '%04d_perturb.npy' % i))).cuda()
                mor_varying_cav(batch_data, results_multi_cav, model, opencood_dataset, adv_feature, cav_max=52)
                time.sleep(0.001)
        except torch.OutOfMemoryError:
            print("Out of memory, skipping this batch")
            torch.cuda.empty_cache()
            continue
        

    for cav_num, result in results_multi_cav.items():
        for metric, result_list in result.items():
            if type(result_list) == list:
                result[metric] = sum(result_list) / len(result_list)
    
    # Convert results_multi_cav into a pandas DataFrame
    processed_results = []
    for cav_num, metrics in results_multi_cav.items():
        row = {'cav_num': cav_num}
        row.update(metrics)
        processed_results.append(row)

    result_stat = pd.DataFrame(processed_results)

    # Save the DataFrame to a CSV file
    df_filename = os.path.join(opt.model_dir, 'diff_cavs.csv')
    result_stat.to_csv(df_filename, index=False)


if __name__ == '__main__':
    main()
