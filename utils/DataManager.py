import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from utils.model_utils import *


def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
):
    """
    Create data loaders for training and testing datasets.

    Args:
        train_dir (str): The directory path of the training dataset.
        test_dir (str): The directory path of the testing dataset.
        transform (torchvision.transforms.Compose): The data transformation to be applied to the datasets.
        batch_size (int): The batch size for the data loaders.

    Returns:
        tuple: A tuple containing the training data loader, testing data loader, and class names.
    """
  
    # Use ImageFolder to create dataset(s)
    train_data = ImageFolder(train_dir, transform=transform)
    test_data = ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    num_workers = os.cpu_count() or 1

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


def data_loader(model_type,train_dir,test_dir):
    print(model_type)
    weights = load_pretrained_model(model_type,True) # .DEFAULT = best available weights from pretraining on ImageNet
    auto_transforms = weights.transforms()
    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
                                                                               batch_size=32) # set mini-batch size to 32
    print(class_names,auto_transforms)
    return train_dataloader, test_dataloader,class_names

def load_data_dir(train_dir, test_dir, batch_size=80, val_size=480):
    train_set = ImageFolder(train_dir, transform=transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor()]))

    test_set = ImageFolder(test_dir, transform=transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor()]))
    
    img, _ = train_set[0]
    print('Size image:', img.shape)
    print('   Classes:', train_set.classes)

    # Split train data into train and validation sets
    train_size = len(train_set) // 2
    train_data, val_data = random_split(train_set, [train_size*2 - val_size*2, val_size*2])

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, num_workers=2, pin_memory=True)


    return (train_dl, val_dl)


def load_data(train_dir, test_dir, batch_size=80, val_size=480):
    # Load train and test data
    train_set = ImageFolder(train_dir, transform=transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor()]))

    test_set = ImageFolder(test_dir, transform=transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor()]))

    # Print image size and classes
    img, _ = train_set[0]
    print('Size image:', img.shape)
    print('   Classes:', train_set.classes)

    # Split train data into train and validation sets
    train_size = len(train_set) // 2
    train_data, val_data = random_split(train_set, [train_size*2 - val_size*2, val_size*2])
    train_set1, train_set2 = random_split(train_set, [train_size, train_size])
    train_data1, val_data1 = random_split(train_set1, [train_size - val_size, val_size])
    train_data2, val_data2 = random_split(train_set2, [train_size - val_size, val_size])

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    train_dl1 = DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    train_dl2 = DataLoader(train_data2, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, num_workers=2, pin_memory=True)
    val_dl1 = DataLoader(val_data1, batch_size=batch_size, num_workers=2, pin_memory=True)
    val_dl2 = DataLoader(val_data2, batch_size=batch_size, num_workers=2, pin_memory=True)


    return [(train_dl, val_dl),(train_dl1, val_dl1),(train_dl2, val_dl2)]

def data_portioning(n):
    #train and test data directory
    train_dir = "/home/dcasals/TFG/datasets/eyenose/train"
    test_dir = "/home/dcasals/TFG/datasets/eyenose/test"

    #load the train and test data
    train_set = ImageFolder(train_dir,transform = transforms.Compose([
        transforms.Resize((32,32)),transforms.ToTensor()]))

    test_set = ImageFolder(test_dir,transforms.Compose([
        transforms.Resize((32,32)),transforms.ToTensor()]))

    batch_size = 80
    val_size   = 480/(n-1)
    portions = [1/n for _ in range(n)]

    train_size = len(train_set) //n
    subsets = random_split(train_set,portions)
    return subsets

def set_servers_data(nodes)->list:
    subsets = list()
    subsets = data_portioning(len(nodes))
    batch_size = 80
    val_size   = 480/(len(nodes)-1)
    dataset = list()
    # train_size = len(train_set) //n
    for idx,node in enumerate(nodes,start=0):
        train_dl, val_dl = random_split(subsets[idx],[0.8,0.2])
        torch.save((train_dl,val_dl),'/home/dcasals/TFG/datasets/eyenose/server_'+node+".pt")
        dataset.append((train_dl,val_dl))
    return dataset

def get_train_test(name):
    t,v =  torch.load('/home/dcasals/TFG/datasets/eyenose/server_'+str(name)+".pt")
    train_dl = DataLoader(t, batch_size=80, shuffle=True, num_workers=1, pin_memory=True)
    val_dl = DataLoader(t, batch_size=80, shuffle=True, num_workers=1, pin_memory=True)
    return train_dl,val_dl




# Example usage:
# print("HOLA")
# train_dir = "eyenose/train"
# test_dir = "eyenose/test"
# batch_size = 80
# val_size = 480

# train_data, val_data = load_data(train_dir, test_dir, batch_size, val_size)



