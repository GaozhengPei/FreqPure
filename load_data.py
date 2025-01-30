import os
import io
import six

from PIL import Image
import lmdb
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import folder, ImageFolder


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)
        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        # load label
        target = unpacked[1]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def imagenet_lmdb_dataset(
        root, transform=None, target_transform=None,
        loader=lmdb_loader):
    """
    You can create this dataloader using:
    train_data = imagenet_lmdb_dataset(traindir, transform=train_transform)
    valid_data = imagenet_lmdb_dataset(validdir, transform=val_transform)
    """
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)
    return data_set





def cifar10_dataset_sub(root, transform=None, num_sub=512, data_seed=0):
    dataset = torchvision.datasets.CIFAR10(
        root=root, transform=transform, download=False, train=False)
    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(
            len(dataset), num_sub, replace=False)
        dataset = Subset(dataset, partition_idx)
    return dataset

def imagenet_dataset_sub(root, transform=None, num_sub=512, data_seed=0):
    print(root)
    dataset = torchvision.datasets.ImageNet(root=root,split='val',transform=transform)
    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(
            len(dataset), num_sub, replace=False)
        dataset = Subset(dataset, partition_idx)
    return dataset


def svhn_dataset_sub(root, transform=None, num_sub=512, data_seed=0):
    dataset = torchvision.datasets.SVHN(
        root=root, split='test', transform=transform, download=True)
    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(
            len(dataset), num_sub, replace=False)
        dataset = Subset(dataset, partition_idx)
    return dataset


def load_cifar10_sub(root, num_sub, data_seed=0):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = cifar10_dataset_sub(
        root, transform=transform, num_sub=num_sub, data_seed=data_seed)
    return dataset


def load_imagenet_sub(root, num_sub, data_seed=0):
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        # transforms.CenterCrop(size=(224,224)),
        transforms.ToTensor(),
    ])
    dataset = imagenet_dataset_sub(
        root, transform=transform, num_sub=num_sub, data_seed=data_seed)
    return dataset


def load_svhn_sub(root, num_sub, data_seed=0):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = svhn_dataset_sub(
        root, transform=transform, num_sub=num_sub, data_seed=data_seed)
    return dataset


def get_imagenet_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def imagenet_lmdb_dataset_sub(root, num_sub=512, data_seed=0):
    transform = get_imagenet_transform()
    dataset = ImageFolderLMDB(root, transform)
    partition_idx = np.random.RandomState(data_seed).choice(
        len(dataset), num_sub, replace=False)
    dataset = Subset(dataset, partition_idx)
    return dataset


def load_dataset_by_name(dataset, root, num_sub):
    if dataset == 'cifar10':
        dataset = load_cifar10_sub(os.path.join(root,dataset), num_sub)
    elif dataset == 'imagenet':
        dataset = load_imagenet_sub(os.path.join(root,dataset), num_sub)
    elif dataset == 'svhn':
        dataset = load_svhn_sub(os.path.join(root,dataset), num_sub)
    print('dataset len: ', len(dataset))
    return dataset
