from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from scipy.io import loadmat
import torch as t
import os
device = t.device('cuda:1' if t.cuda.is_available() else 'cpu')



# 读取数据集图片
class MyDataset(Dataset):
    def __init__(self, root, mode, alpha):
        super(MyDataset, self).__init__()
        self.root = root
        self.mode = mode
        self.alpha = alpha
        self.transform_1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.RandomRotation(180),
        ])
        if mode == 'test':
            # 建立索引 + 对索引排序
            self.lrHSroot = os.listdir(os.path.join(root, "test", "LRHS"+ alpha))
            self.lrHSroot.sort(key=lambda x: int(x.split(".")[0]))
            self.PANroot = os.listdir(os.path.join(root, "test", "hrMS"))
            self.PANroot.sort(key=lambda x: int(x.split(".")[0]))
            self.gtHSroot = os.listdir(os.path.join(root, "test", "gtHS"))
            self.gtHSroot.sort(key=lambda x: int(x.split(".")[0]))

        if mode == 'train':
            self.lrHSroot = os.listdir(os.path.join(root, "train", "LRHS"+ alpha))
            self.lrHSroot.sort(key=lambda x: int(x.split(".")[0]))
            self.PANroot = os.listdir(os.path.join(root, "train", "hrMS"))
            self.PANroot.sort(key=lambda x: int(x.split(".")[0]))
            self.gtHSroot = os.listdir(os.path.join(root, "train", "gtHS"))
            self.gtHSroot.sort(key=lambda x: int(x.split(".")[0]))

    def __getitem__(self, item):
        LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS"+ self.alpha, self.lrHSroot[item]))['LRHS'].reshape(-1, 40, 40)
        PAN = loadmat(os.path.join(self.root, self.mode, "hrMS", self.PANroot[item]))['hrMS'].reshape(-1, 160, 160)
        gtHS = loadmat(os.path.join(self.root, self.mode, "gtHS", self.gtHSroot[item]))['gtHS'].reshape(-1, 160, 160)

        return LRHS, PAN, gtHS

    def __len__(self):
        return self.gtHSroot.__len__()


if __name__ == "__main__":
    root = "../dataset/pavia"
    data = MyDataset(root, "train", '_alpha1')
    print(data.__len__())
    print(data.__getitem__(0))