import os
import torch
from adamp import AdamP
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
import datetime as datetimes
import time as times
import sys
from util import *
from ops import *
import shutil
from tqdm import tqdm

time = datetimes.datetime.now().strftime('%m.%d-%H:%M:%S')

class Solver():
    def __init__(self, model, cfg):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.refiner = model(scale=cfg.scale,n_blocks = cfg.n_blocks)

        if cfg.loss_fn in ["MSE"]:
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]:
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = AdamP(
            filter(lambda p: p.requires_grad, self.refiner.parameters()),
            cfg.lr)

        self.train_data = TrainDataset(cfg.train_data_path,
                                       scale=cfg.scale,
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       pin_memory=True,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)

        self.refiner = self.refiner.to(self.device)
        self.loss_fn = self.loss_fn
        self.loss_fn_L2 = nn.MSELoss()
        self.folder_name = str(cfg.loss_fn) + '_' + str(cfg.batch_size) + '_' + str(cfg.max_steps)[0] + 'K' + '_' + \
                           str(cfg.lr) + '_x' + str(cfg.scale) + str(cfg.end_fix)

        checkpoint_folder = 'logs/{}/checkpoints'.format(self.folder_name)
        mkdir(checkpoint_folder)

        if cfg.resume:
            PATH = os.path.join("logs", self.folder_name, "checkpoints")
            all_checkpoints = list(sorted(os.listdir(PATH)))

            if len(all_checkpoints) > 0:
                PATH = os.path.join(PATH, all_checkpoints[-1])
                print("=> loading checkpoint '{}'".format(PATH))
                checkpoint = torch.load(PATH)
                self.refiner.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                self.step = checkpoint['step']
                self.best_psnr = checkpoint["best_psnr"]
            else:
                print("=> no checkpoint at '{}'".format(PATH))
                self.best_psnr = 0
                self.step = 0
        else:
            self.best_psnr = 0
            self.step = 0

        self.cfg = cfg

        self.writer = SummaryWriter(log_dir=os.path.join("logs/{}/tensorboard/".format(self.folder_name)))
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("Number of parameters for scale X{}: {}".format(cfg.scale, num_params))


    def fit(self):
        cfg = self.cfg

        refiner = nn.DataParallel(self.refiner,
                                  device_ids=range(cfg.num_gpu))
        self.mean_l1 = 0.

        learning_rate = cfg.lr

        with tqdm(total=cfg.max_steps) as t:
            t.update(self.step)
            while True:
                for inputs in self.train_loader:

                    self.refiner.train()

                    scale = cfg.scale

                    hr, lr = inputs[-1][0], inputs[-1][1]

                    hr = hr.to(self.device)
                    lr = lr.to(self.device)

                    sr_main = refiner(lr)

                    loss = self.loss_fn(sr_main, hr)

                    self.optim.zero_grad()
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                    self.optim.step()

                    self.mean_l1 += loss

                    learning_rate = self.decay_learning_rate()
                    for param_group in self.optim.param_groups:
                        param_group["lr"] = learning_rate

                    self.step += 1
                    
                    t.update(1)
                    t.set_description('Steps: {}/{}'.format(self.step, cfg.max_steps))
                    t.set_postfix(loss='{:.6f}'.format(loss.item()),lr='{:1.4}'.format(learning_rate))

                    self.writer.add_scalar('Loss', loss.data.cpu().numpy(), global_step=self.step)

                    if cfg.verbose and self.step % cfg.print_interval == 0:
                        with open('logs/{}/'.format(self.folder_name) + 'logs.txt', 'a') as f:
                            PATH = os.path.join('logs/{}/checkpoints/'.format(self.folder_name),
                                                "{}_{:06d}.pth.tar".format(cfg.ckpt_name, self.step))

                            mean_psnr = self.evaluate(cfg.valid_data_path, scale=cfg.scale, num_step=self.step)

                            self.writer.add_scalar("PSNR_{}x:".format(scale), mean_psnr, self.step)
                            self.writer.add_scalar("1000mean_loss_x{}:".format(scale), (self.mean_l1) / cfg.print_interval, self.step)

                            print('-- PSNR_x{}: {:.5f}  -- Total_Loss: {:.5f}\n'
                                            .format(scale, mean_psnr,  (self.mean_l1) / cfg.print_interval))

                            torch.save({'step': self.step, 'model_state_dict': self.refiner.state_dict(),
                                            'optimizer_state_dict': self.optim.state_dict(), 'best_psnr': self.best_psnr}, PATH)

                            f.write('Step: {}'
                                                '--> PSNR_x{}:{:.5f} \n'
                                                    .format(self.step, scale, mean_psnr))

                        self.mean_l1 = 0.


                if self.step > cfg.max_steps: break

    def evaluate(self, test_data_dir, scale=2, num_step=0):
        cfg = self.cfg
        mean_psnr = 0

        self.refiner.eval()

        test_data = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data, batch_size=1, num_workers=1, shuffle=True)

        for step, inputs in enumerate(test_loader):
            hr = inputs[0]
            lr = inputs[1]
            name = inputs[2][0]

            lr = lr.to(self.device)
            hr = hr.to(self.device)
            # run refine process in here!

            sr = self.refiner(lr)

            psnr = calc_psnr(sr, hr, scale, 1, benchmark=True)
            mean_psnr += psnr / len(test_data)


        return mean_psnr


    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_name):
        save_path = os.path.join(
            'logs/{}/checkpoints/'.format(self.folder_name), "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay+self.step // 200000))
        return lr

    def save_checkpoint(self, is_best, filename='checkpoint.pth.tar'):
        save_path = os.path.join(self.cfg.logdir, self.folder_name) + '/'
        torch.save(self.refiner, save_path + filename)
        if is_best:
            shutil.copyfile(save_path + filename, save_path + 'model_best.pth.tar')
