import numpy as np
from torch.utils.data import Dataset
import torch
import cv2


class GXHDataset(Dataset):
    def __init__(self, path='./data', t='train'):
        """Init function."""
        self.data_50 = np.load(f"{path}/crop50_dataset_gyh.npy")
        self.data_20 = np.load(f"{path}/crop20_dataset_gyh.npy")

        data_crop_len = len(self.data_50)
        _ = int(data_crop_len * 0.8)
        __ = int(data_crop_len * 0.9)

        # self.data_50 = np.clip(self.data_50, 0, 100).astype(np.float32) / 100.0
        # self.data_20 = np.clip(self.data_20, 0, 100).astype(np.float32) / 100.0

        if t == 'train':
            self.data_50 = self.data_50[:_]
            self.data_20 = self.data_20[:_]
        elif t == 'test':
            self.data_50 = self.data_50[_:__]
            self.data_20 = self.data_20[_:__]
        else:
            self.data_50 = self.data_50[__:]
            self.data_20 = self.data_20[__:]


    def __getitem__(self, index):
        """Get item."""
        crop50_seq = self.data_50[index]
        crop20_seq = self.data_20[index]
        # 值范围 0-100
        return torch.from_numpy(np.array(crop50_seq, dtype=np.float32)), torch.from_numpy(np.array(crop20_seq, dtype=np.float32))

    def __len__(self):
        """Length."""
        return len(self.data_50)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    def save_img(img_data, image_path):
        # 获取数组尺寸
        h, w = np.array(img_data).shape
        # 创建无框画布
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w / 100, h / 100)
        ax = plt.Axes(fig, [0, 0, 1, 1])  # 铺满整个画布
        ax.set_axis_off()
        fig.add_axes(ax)
        # 绘图
        # v_min = np.percentile(img_data, 1)  # 取1%分位
        # v_max = np.percentile(img_data, 99)  # 取99%分位
        # ax.imshow(img_data, aspect='auto', vmin=v_min, vmax=v_max)
        ax.imshow(img_data, aspect='auto')
        # 保存：dpi=100 保证 图片尺寸 = 数组尺寸（w像素 × h像素）
        fig.savefig(image_path, dpi=100)
        plt.close()

    data_set = GXHDataset('./data', t='train')
    dataLoader = torch.utils.data.DataLoader(data_set, batch_size=32, shuffle=True, drop_last=True)

    print(len(dataLoader))

    count = 0  # 计数器
    for dx, dy in dataLoader:
        if count >= 1:  # 只打印前10个
            break
        print('-------------------')
        print(dx.shape)
        print(dx.dtype)
        print(dy.shape)
        print(dy.dtype)
        count += 1  # 每循环一次+1

        for i, (scan_data_A, scan_data_B) in enumerate(zip(dx, dy)):
            for j, (sda, sdb) in enumerate(zip(scan_data_A, scan_data_B)):
                print(sda.max(), sda.min(), sdb.max(), sdb.min())
                save_img(sda, f'./原始数据/75MHz - 30dB - crop/{i}_{j}_50μm.png')
                save_img(sdb, f'./原始数据/75MHz - 30dB - crop/{i}_{j}_20μm.png')
