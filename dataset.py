import numpy as np
from torch.utils.data import Dataset
import torch


class GXHDataset(Dataset):
    def __init__(self, path='./data', t='train'):
        """Init function."""
        self.data_50 = np.load(f"{path}/crop50_dataset_gyh.npy")
        self.data_20 = np.load(f"{path}/crop20_dataset_gyh.npy")

        data_crop_len = len(self.data_50)
        _ = int(data_crop_len * 0.8)

        if t == 'train':
            self.data_50 = self.data_50[:_]
            self.data_20 = self.data_20[:_]
        else:
            self.data_50 = self.data_50[_:]
            self.data_20 = self.data_20[_:]


    def __getitem__(self, index):
        """Get item."""
        crop50_seq = self.data_50[index]
        crop20_seq = self.data_20[index]
        # 值范围 0-100
        return torch.from_numpy(np.array(crop50_seq, dtype=np.float32)), torch.from_numpy(np.array(crop20_seq, dtype=np.float32))

    def __len__(self):
        """Length."""
        return len(self.data_50)
