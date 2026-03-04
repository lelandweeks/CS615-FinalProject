import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def LoadData(color=True, transforms=transforms.ToTensor(), batch_size=32, shuffle=True, num_workers=2):

    # choose color or grayscale 
    if color:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    data_folder = '../images'
    
    train_dataset = datasets.ImageFolder(os.path.join(f'{data_folder}', 'train'), transform=transforms)
    test_dataset = datasets.ImageFolder(os.path.join(f'{data_folder}', 'test'), transform=transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader, train_dataset, test_loader, test_dataset