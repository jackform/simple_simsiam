from PIL import Image
from torch.utils.data import Dataset
import os


class FoodDataset(Dataset):

    def __init__(self, train=True, tfm=None, needDoubleTransformImage=True, needLabel=True):
        super(FoodDataset).__init__()
        _dataset_dir = "../input/ml2022spring-hw3b/food11"

        if needLabel and needDoubleTransformImage:
            path = os.path.join(_dataset_dir, "test")
        else:
            if train:
                path = os.path.join(_dataset_dir, "training")
            else:
                path = os.path.join(_dataset_dir, "validation")

        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        print(f"One sample", self.files[0])
        self.transform = tfm
        self.size = len(self.files)
        self.classes = [str(i + 1) for i in range(11)]
        # self.classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        self.train = train
        # self.targets = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(self.size)]
        self.targets = [i for i in range(11)]
        self.needDoubleTransformImage = needDoubleTransformImage

    def __len__(self):
        return len(self.files)
        # return 8

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        # print('===========>', 'needDoubleImage:', self.needDoubleTransformImage)
        if self.needDoubleTransformImage:
            im1, im2 = self.transform(im)
            return (im1, im2), label
        else:
            im = self.transform(im)
            return im, label
