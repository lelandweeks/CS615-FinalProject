import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def LoadMetadata():
    data_folder = 'csv'
    all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
    df_list = [pd.read_csv(filename) for filename in all_files]
    return df_list

def LoadImages(color=True, batch_size=32, shuffle=True):


    # choose color or grayscale 
    if color:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    data_folder = 'images'
    
    train_dataset = datasets.ImageFolder(os.path.join(f'{data_folder}', 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(f'{data_folder}', 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, train_dataset, test_loader, test_dataset