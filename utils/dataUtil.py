import numpy as np
import torchvision
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms


# Data
def getDataLoader(split_factor: float = 0.1, seed: int = 66, data_set: str = 'CIFAR10'):
    if data_set == 'CIFAR10':
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        train_Transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_Transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_set = torchvision.datasets.CIFAR10(
            root='/home/lab265/lab265/datasets/CIFAR10', train=True, download=True, transform=train_Transforms)

        test_set = torchvision.datasets.CIFAR10(
            root='/home/lab265/lab265/datasets/CIFAR10', train=False, download=True, transform=test_Transforms)
    elif data_set == 'CIFAR100':
        print('=> loading cifar100 data...')
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        normalize = transforms.Normalize(mean=mean, std=std)
        train_Transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_Transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_set = torchvision.datasets.CIFAR100(
            root='/home/lab265/lab265/datasets/CIFAR100', train=True, download=True, transform=train_Transforms)

        test_set = torchvision.datasets.CIFAR100(
            root='/home/lab265/lab265/datasets/CIFAR100', train=False, download=True, transform=test_Transforms)

    dataset_size = len(train_set)
    indices = list(range(dataset_size))
    split = int(np.floor(split_factor * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_set, batch_size=128, sampler=train_sampler,
                              num_workers=4, drop_last=False, pin_memory=True)
    validation_loader = DataLoader(train_set, batch_size=100, sampler=valid_sampler,
                                   num_workers=4, drop_last=False,
                                   pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
    return train_loader, validation_loader, test_loader
