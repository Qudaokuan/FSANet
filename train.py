import os
import json
import argparse
import importlib
from solver import Solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='FESNet')
    parser.add_argument("--ckpt_name", type=str)

    parser.add_argument("--print_interval", type=int, default=1000)
    parser.add_argument("--train_data_path", type=str, default="/root/shiyan/FENet/datasets/DIV2K_train.h5")
    parser.add_argument("--valid_data_path", type=str, default="/share/datasets/DIV2K/test/M/Set5")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint")
    parser.add_argument("--sample_dir", type=str, default="sample")

    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--pre_trained", type=bool, default=False)

    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--scale", type=int, default=3)

    parser.add_argument("--verbose", action="store_true", default="store_true")

    parser.add_argument("--group", type=int, default=1)

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=64)

    parser.add_argument("--max_steps", type=int, default=600000)
    parser.add_argument("--decay", type=int, default=150000)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--clip", type=float, default=0.2)

    parser.add_argument("--loss_fn", type=str, choices=["MSE", "L1", "SmoothL1", 'L1_Charb'], default="L1")

    parser.add_argument("--gpu_index",type=str,required=True)
    parser.add_argument("--end_fix",type=str,required=True)
    parser.add_argument("--n_blocks",type=int,default=1)
    return parser.parse_args()

def main(cfg):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_index
    net = importlib.import_module("{}".format(cfg.model)).FESNet
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    solver = Solver(net, cfg)
    solver.fit()

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
