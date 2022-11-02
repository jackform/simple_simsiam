import torch
import numpy as np


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, transform, root=None, train=True, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.size = 8
        self.classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        self.train = train
        self.targets = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(self.size)]

    def __getitem__(self, idx):
        # print('======>train:', self.train)
        if idx < self.size:
            if self.train:
                # return [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))], [0, 0, 0]
                # return [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                return [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))], torch.randn(11)
            else:
                # return torch.randn((3, 224, 224)), [0, 0, 0]
                # return torch.randn((3, 224, 224)), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                return torch.randn((3, 224, 224)), torch.randn(11)
        else:
            raise Exception

    def __len__(self):
        return self.size
