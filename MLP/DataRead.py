import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from cutout import Cutout

# 检测可用的设备并设置为cuda或cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 用于数据加载的子进程数量
num_workers = 0

# 每批加载图像的数量
batch_size = 16

# 用于验证集的训练集比例
valid_size = 0.2

def read_dataset(batch_size=16, valid_size=0.2, num_workers=0, pic_path='dataset',dataset = "cifar10"):
    # 定义训练数据的转换操作
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 在图像周围填充0，然后随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 对图像进行标准化
        Cutout(n_holes=1, length=16),  # 随机遮挡图像的一部分
    ])

    # 定义测试数据的转换操作
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载训练、验证和测试数据集
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_train)
        valid_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_test)
        test_data = datasets.CIFAR10(pic_path, train=False, download=True, transform=transform_test)
    elif  dataset == "cifar100":
        train_data = datasets.CIFAR100(pic_path, train=True, download=True, transform=transform_train)
        valid_data = datasets.CIFAR100(pic_path, train=True, download=True, transform=transform_test)
        test_data = datasets.CIFAR100(pic_path, train=False, download=True, transform=transform_test)

    # 获取用于验证的训练数据索引
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # 定义用于训练和验证的采样器
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # 准备数据加载器（组合数据集和采样器）
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
