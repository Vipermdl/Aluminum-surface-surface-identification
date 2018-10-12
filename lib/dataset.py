#encoding:utf-8
from PIL import Image
from torch.utils.data import Dataset



def default_loader(path):
    # return Image.open(path)
    return Image.open(path).convert('RGB')


class TrainDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform()(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class ValDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform()(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class TestDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform()(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)




