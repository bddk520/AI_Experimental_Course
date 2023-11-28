import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from cutout import Cutout

# �����õ��豸������Ϊcuda��cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# �������ݼ��ص��ӽ�������
num_workers = 0

# ÿ������ͼ�������
batch_size = 16

# ������֤����ѵ��������
valid_size = 0.2

def read_dataset(batch_size=16, valid_size=0.2, num_workers=0, pic_path='dataset',dataset = "cifar10"):
    # ����ѵ�����ݵ�ת������
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # ��ͼ����Χ���0��Ȼ������ü���32*32
        transforms.RandomHorizontalFlip(),  # ���ˮƽ��תͼ��
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ��ͼ����б�׼��
        Cutout(n_holes=1, length=16),  # ����ڵ�ͼ���һ����
    ])

    # ����������ݵ�ת������
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ����ѵ������֤�Ͳ������ݼ�
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_train)
        valid_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_test)
        test_data = datasets.CIFAR10(pic_path, train=False, download=True, transform=transform_test)
    elif  dataset == "cifar100":
        train_data = datasets.CIFAR100(pic_path, train=True, download=True, transform=transform_train)
        valid_data = datasets.CIFAR100(pic_path, train=True, download=True, transform=transform_test)
        test_data = datasets.CIFAR100(pic_path, train=False, download=True, transform=transform_test)

    # ��ȡ������֤��ѵ����������
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # ��������ѵ������֤�Ĳ�����
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # ׼�����ݼ�������������ݼ��Ͳ�������
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
