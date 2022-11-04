from PIL import Image
from torch.utils.data import Dataset
import os


class FoodDataset(Dataset):

    def __init__(self, train=True, tfm=None, needDoubleTransformImage=True, needLabel=True):
        super(FoodDataset).__init__()
        _dataset_dir = "/kaggle/input/ml2022spring-hw3b/food11"
    
        if not needLabel and needDoubleTransformImage:
            path = os.path.join(_dataset_dir, "test")
            test_file = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
            path = os.path.join(_dataset_dir, "training")
            train_file = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
            path = os.path.join(_dataset_dir, "validation")
            validate_file = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
            self.files = test_file + train_file + validate_file
        else:
            if train:
                path = os.path.join(_dataset_dir, "training")
            else:
                path = os.path.join(_dataset_dir, "validation")
            self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])

        print(f"One sample", self.files[0])
        self.transform = tfm
        self.size = len(self.files)
        self.classes = [str(i) for i in range(11)]
        # self.classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        self.train = train
        # self.targets = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(self.size)]
        self.needDoubleTransformImage = needDoubleTransformImage
        if needLabel:
            self.targets = [int(fname.split("/")[-1].split("_")[0]) for fname in self.files]
        else:
            self.targets = [-1 for fname in self.files]


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
