import argparse
import os
import random

import numpy as np
import torch

from data_provider.read_geodataframe import load_dataset
from exp.exp_main import Exp_Main


def main():
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Crowd Prediction")

    # exp config
    parser.add_argument("--path", type=str, default="/content/GTFormer", help="current directory")
    parser.add_argument("--model", type=str, default="GTFormer", help="model name")
    parser.add_argument("--sample_time", type=str, default="60min", help="sample time")

    parser.add_argument("--itrs", type=int, default=10, help="number of run")
    parser.add_argument("--train_epochs", type=int, default=50, help="epochs")
    parser.add_argument("--patience", type=int, default=5, help="patience of early stopping")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--seq_len", type=int, default=11, help="input sequence length")
    parser.add_argument("--lr", type=int, default=1e-03, help="learning rate")
    parser.add_argument("--save_outputs", type=bool, default=False, help="save")
    parser.add_argument("--city", type=str, default="NYC", help="city name")
    parser.add_argument("--num_tiles", type=int, default=47, help="number of tiles")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout late")

    # GTFormer config
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--temporal_num_layers", type=int, default=2)
    parser.add_argument("--spatial_num_layers", type=int, default=1)
    parser.add_argument("--use_relativepos", type=bool, default=True, help="BRPE")
    parser.add_argument("--use_kvr", type=bool, default=True, help="KVR")
    parser.add_argument("--use_only", type=bool, default="None", help='["Spatial", "Temporal", "None"]')

    # CrowdNet config
    parser.add_argument("--d_temporal", type=int, default=64)
    parser.add_argument("--d_spatial", type=int, default=16)

    args = parser.parse_args(args=[])

    dataset_directory = os.path.join(args.path + "/data/" + args.city + "/")
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)
    if not os.path.isfile(dataset_directory + "df_grouped_1000m_" + args.sample_time + ".csv"):
        load_dataset(args.city, args.sample_time, dataset_directory)

    print("Args in experiment:")
    print(args)

    Exp = Exp_Main

    for itr in range(args.itrs):
        print("\n")
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print(f"itr : {itr+1}")

        exp = Exp(args)  # set experiments
        print(">>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train()

        print(">>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(itr)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
