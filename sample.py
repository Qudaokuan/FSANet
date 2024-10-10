import json
import time
import importlib
import argparse
import torch
from dataset import TestDataset
from util import *
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='FESNet')
    parser.add_argument("--ckpt_path", type=str, default='./logs/checkpoint_x4/None_583000.pth.tar')
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--sample_dir", type=str, default='sample')
    parser.add_argument("--test_data_dir", type=str, default="./dataset/DIV2K/Set5")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--n_blocks",type=int,default=3)
    parser.add_argument("--gpu_index",type=str,default='0')

    return parser.parse_args()

def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def sample(net, dataset, cfg):
    avg_psnr = 0
    avg_ssim = 0
    cuda = True if torch.cuda.is_available() else False
    scale = cfg.scale
    for step, (hr, lr, name) in enumerate(dataset):
        # if "Set5" not in dataset.name:
        t1 = time.time()

        h, w = lr.size()[1:]
        h_half, w_half = int(h / 2), int(w / 2)
        h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

        # split large image to 4 patch to avoid OOM error
        lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
        lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
        lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
        lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
        lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])
        lr_patch = lr_patch.cuda()

        # run refine process in here!
        with torch.no_grad():
            sr = net(lr_patch)

        h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
        w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

        # merge splited patch images
        result = torch.FloatTensor(3, h, w).cuda()
        result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
        result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
        result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
        result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
        sr = result

        t2 = time.time()

        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        sr_dir = os.path.join(cfg.sample_dir, model_name,
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "SR")

        hr_dir = os.path.join(cfg.sample_dir,model_name, cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "HR")

        os.makedirs(sr_dir, exist_ok=True)
        os.makedirs(hr_dir, exist_ok=True)

        sr_im_path = os.path.join(sr_dir, "{}".format(name.replace("HR", "SR")))
        hr_im_path = os.path.join(hr_dir, "{}".format(name))

        save_image(sr, sr_im_path)

        sr = sr.unsqueeze(0).cuda()
        hr = hr.unsqueeze(0).cuda()

        psnr = calc_psnr(sr, hr, scale, 1, benchmark=True)
        ssim = calc_ssim(sr,hr,scale)
        avg_psnr += psnr / len(dataset)
        avg_ssim += ssim / len(dataset)

    GREEN = '\033[92m'
    END_COLOR = '\033[0m'
    print(GREEN+'Average PSNR/SSIM on scale X{} is {:.2f}/{:.4f}'.format(cfg.scale, avg_psnr,avg_ssim)+END_COLOR)


def main(cfg):
    module = importlib.import_module("{}".format(cfg.model))
    net = module.FESNet(multi_scale=False, group=cfg.group, scale=cfg.scale,n_blocks = cfg.n_blocks)
    checkpoint = torch.load(cfg.ckpt_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.cuda()

    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    sample(net, dataset, cfg)


if __name__ == "__main__":
    cfg = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_index
    main(cfg)


