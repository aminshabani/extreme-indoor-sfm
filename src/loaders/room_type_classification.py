import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from .main_loader import load_dataset
import logging
import json
from PIL import Image
from glob import glob
import src.panotools

# return  img1, img2, mask1, mask2, label, rtype1, rtype2
ROOM_TYPES = ['Western-style_room', 'Entrance', 'Kitchen', 'Verandah',
              'Balcony', 'Toilet', 'Washing_room', 'Bathroom', 'Japanese-style_room']

logger = logging.getLogger('log')
BASE_DIR = ''


class FloorDataset(Dataset):
    def __init__(self, set_name, transform, retorg=False):
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.CLAHE(),
            A.ISONoise(),
        ])
        logger.info("Loading data from {} ...".format(BASE_DIR))
        self.ret_org = retorg
        self.transforms = transform
        self.set_name = set_name
        self.samples = []
        # self.houses = load_dataset()
        # TYPES: [Balcony, Closet, Western style room, Japanese style room, Dining Room
        #  Kitchen, Corridor, Washroom, Bathroom, Toilet]
        names = [line.rstrip() for line in open('data_names.txt')]
        test_names = [line.rstrip() for line in open('test_names.txt')]

        train_houses = []
        test_houses = []
        for name in names:
            if name not in test_names:
                train_houses.append(panotools.House(BASE_DIR, name))
            else:
                test_houses.append(panotools.House(BASE_DIR, name))
        if set_name == 'train':
            for house in train_houses:
                for pano in house.panos:
                    if(pano.get_type()==-1):
                        continue
                    self.samples.append([pano, pano.get_type()])
        else:
            for house in test_houses:
                for pano in house.panos:
                    if(pano.get_type()==-1):
                        continue
                    self.samples.append([pano, pano.get_type()])


        per_class_samples = np.zeros(10)
        for sample in self.samples:
            assert sample[1] < 10, sample
            per_class_samples[sample[1]] += 1
        print(per_class_samples, len(self.samples))

        logger.info(per_class_samples)
        logger.info("finish loading with {} houses and {} samples...".format(
            1000, len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        pano = sample[0]
        label = sample[1]
        img = pano.get_panorama()
        img = np.array(img)

        if(self.set_name == 'train'):
            transformed = self.augmentation(image=img)
            img = transformed['image']
        img = img.astype(float)/255

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


def dataset(set_name='train', batch_size=2, house_id=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if (set_name == 'train'):
        dataset = FloorDataset(set_name, transform=transform)
    else:
        dataset = FloorDataset(set_name, transform=transform)

    if (set_name == 'train'):
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=10,
                            pin_memory=True,
                            shuffle=True, prefetch_factor=10)
    else:
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=10,
                            pin_memory=True,
                            shuffle=False)

    return loader


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = FloorDataset('train', transform=transform)
    for i in range(10):
        dataset[i]
