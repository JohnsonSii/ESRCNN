import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class MyDataset(Dataset):
    def __init__(self, root, file, width, height):
        self.imgs = []
        self.width = width
        self.height = height
        with open(os.path.join("..", "data", file)) as rf:
            for line in rf:
                line = line.replace("\n", "")
                self.imgs.append(os.path.join(root, line + ".png"))

    def __getitem__(self, item):
        img = Image.open(self.imgs[item])
        img = transforms.ToTensor()(img)
        img_crop = transforms.RandomCrop((self.width, self.height))(img)

        data = transforms.ToPILImage()(img_crop).resize((64, 64))
        data = transforms.ToTensor()(data)

        # data = transforms.Resize(32)(img_crop)
        # data = img.resize((int(self.width / 2), int(self.height / 2)))
        # label = transforms.Resize(256)(img_crop)
        # target = np.array(label).astype('int32')

        return data, img_crop

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    myDataset = MyDataset(r"E:\Dataset\DIV2K_train_HR\DIV2K_train_HR", "train.txt", 512, 512)
    for data, label in myDataset:
        data_img = transforms.ToPILImage()(data)
        data_img.show()
        print(data.size())

        label_img = transforms.ToPILImage()(label)
        label_img.show()
        print(label.size())

        break
