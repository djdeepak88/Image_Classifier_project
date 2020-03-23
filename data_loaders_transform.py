import os
import torch
from torchvision import datasets, transforms
import torchvision
from torch import optim

def get_data_loaders(data_path):
    """
    Creating the data loaders from the train, valid and test_dir
    from the data_path.
    """
    train_dir = data_path + '/train'
    valid_dir = data_path + '/valid'
    test_dir = data_path + '/test'

    data_transforms = get_data_transforms()

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(valid_dir, transform=data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    data_loaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    return image_datasets, data_loaders


def get_data_transforms():

    """
    Data Transformation.
    """

    return {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                     ]
                                    ),
        'val': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                   ]
                                  ),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ]
                                   )
    }
