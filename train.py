from torch import optim
from dataset import GXHDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from module4 import *
from history import History
import torch.nn as nn
import numpy as np

import time


def parse_args():
    # PARAMETERS
    parser = argparse.ArgumentParser('DASR-Net')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers in training')

    return parser.parse_args()

def train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device):

    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_grad = 0.0
    total_depth = 0.0
    total_lap = 0.0

    total_psnr = 0.0

    for lr_img, hr_img in train_loader:

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        optimizer.zero_grad()

        pred = model(lr_img)

        loss_dict = criterion(
            pred,
            hr_img
        )

        # ==========================
        # batch loss -> scalar
        # ==========================

        loss = loss_dict['total'].mean()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        total_l1 += loss_dict['l1'].mean().item()

        total_grad += loss_dict['grad'].mean().item()

        total_depth += loss_dict['depth'].mean().item()

        total_lap += loss_dict['lap'].mean().item()

        mse = F.mse_loss(
            pred,
            hr_img
        )

        psnr = 20 * torch.log10(
            1.0 / torch.sqrt(mse)
        )

        total_psnr += psnr.item()

    n = len(train_loader)

    return {
        'loss': total_loss / n,
        'l1': total_l1 / n,
        'grad': total_grad / n,
        'depth': total_depth / n,
        'lap': total_lap / n,
        'psnr': total_psnr / n
    }

@torch.no_grad()
def evaluate(
        model,
        test_loader,
        criterion,
        device):

    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_grad = 0.0
    total_depth = 0.0
    total_lap = 0.0
    total_psnr = 0.0

    for lr_img, hr_img in test_loader:

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        pred = model(lr_img)

        loss_dict = criterion(
            pred,
            hr_img
        )

        total_loss += loss_dict['total'].mean().item()

        total_l1 += loss_dict['l1'].mean().item()

        total_grad += loss_dict['grad'].mean().item()

        total_depth += loss_dict['depth'].mean().item()

        total_lap += loss_dict['lap'].mean().item()

        mse = F.mse_loss(
            pred,
            hr_img
        )

        psnr = 20 * torch.log10(
            1.0 / torch.sqrt(mse)
        )

        total_psnr += psnr.item()

    n = len(test_loader)

    return {
        'loss': total_loss / n,
        'l1': total_l1 / n,
        'grad': total_grad / n,
        'depth': total_depth / n,
        'lap': total_lap / n,
        'psnr': total_psnr / n
    }

def main(args):
    # 记录程序开始时间
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    print(device)
    print(torch.cuda.is_available())

    # 读取数据
    data_path = './data'

    train_dataset = GXHDataset(data_path, t='train')
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    test_dataset = GXHDataset(data_path, t='test')
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # 实例化模型
    # m = DASRNet(6, 64).to(device)

    # 多GPU板卡
    m = DASRNet(6, 64)

    m = torch.nn.DataParallel(m, device_ids=[0, 1])

    m = m.to(device)

    criterion = TotalLoss().to(device)

    info = ''
    h = History('DASR-Net', './history_save', args.learning_rate, args.epoch, args.batch_size, info=info)

    # 优化器选择和动态学习率
    optimizer = optim.Adam(m.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # 训练模型
    best_psnr = 0.0

    for epoch in range(1, 201):

        start_time_epoch = time.time()

        train_log = train_one_epoch(
            m,
            train_data_loader,
            criterion,
            optimizer,
            device
        )

        test_log = evaluate(
            m,
            test_data_loader,
            criterion,
            device
        )

        scheduler.step()

        elapsed = time.time() - start_time_epoch

        print(
            f'Epoch [{epoch:03d}/200] | Time: {elapsed:.2f}s\n'
            f'Train Loss : {train_log["loss"]:.6f}\n'
            f'  L1        : {train_log["l1"]:.6f}\n'
            f'  Grad      : {train_log["grad"]:.6f}\n'
            f'  Depth     : {train_log["depth"]:.6f}\n'
            f'PSNR        : {train_log["psnr"]:.4f} dB\n'
            f'Test Loss  : {test_log["loss"]:.6f}\n'
            f'  L1        : {test_log["l1"]:.6f}\n'
            f'  Grad      : {test_log["grad"]:.6f}\n'
            f'  Depth     : {test_log["depth"]:.6f}\n'
            f'PSNR        : {test_log["psnr"]:.4f} dB'
        )

        if test_log['psnr'] > best_psnr:
            best_psnr = test_log['psnr']

            torch.save(m.module.cpu().state_dict(), './model_save/DASR-Net_best.pth'.format(info))

            print(
                f'Best model saved. '
                f'PSNR={best_psnr:.4f}'
            )

        # 保存网络模型 多板卡 使用 m.module
        torch.save(m.module.cpu().state_dict(), './model_save/DASR-Net_last.pth'.format(info))
        m = m.to(device)

        h.train_loss.append(train_log)
        h.test_loss.append(test_log)
        h.save_history()


    # 保存网络模型
    torch.save(m.module.cpu().state_dict(), './model_save/DASR-Net_last.pth'.format(info))

    # 记录程序结束时间
    end_time = time.time()

    # 计算运行时间，单位为秒
    execution_time = end_time - start_time
    print(f"程序运行时间：{execution_time} 秒")

    h.execution_time = execution_time
    h.save_history()


if __name__ == '__main__':
    a = parse_args()
    main(a)
