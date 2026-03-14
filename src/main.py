import argparse
import torch

import data
import evaluate
import model
import train

"""
 EXPERIMENTS:
 * color vs grayscale
 * 1 kernel vs multiple kernels
 * 3x3 vs 5x5 kernel size
 * 1 conv layer vs multiple conv layers

BASE MODEL: Color images with 1 3x3 kernel and 1 convolutional layer
    PARAMETERS: in_channels=3, num_kernels=1, kernel_size=3, num_conv_layers=1
    FILE NAME: model_color_1k_3x3_1l.pt

2D MODEL: Same as base with grayscale images
    PARAMETERS: in_channels=1, num_kernels=1, kernel_size=3, num_conv_layers=1
    FILE NAME: model_gray_1k_3x3_1l.pt
 
MULTIPLE KERNELS MODEL: Same as base with multiple kernels
    PARAMETERS: in_channels=3, num_kernels=3, kernel_size=3, num_conv_layers=1
    FILE NAME: model_color_3k_3x3_1l.pt
 
KERNEL SIZE MODEL: Same as base with 5x5 kernels
    PARAMETERS: in_channels=3, num_kernels=1, kernel_size=5, num_conv_layers=1
    FILE NAME: model_color_1k_5x5_1l.pt

MULTIPLE CONV LAYERS MODEL: Same as base with multiple convolutional layers
    PARAMETERS: in_channels=3, num_kernels=1, kernel_size=3, num_conv_layers=3
    FILE NAME: model_color_1k_3x3_3l.pt
"""

def main():

    # parse the command line arguments to target the corresponding model
    args = parse_args()
    print(args)
    color = True
    dims = 'color'
    if args.color == False:
        color
        dims = 'gray'
    kernels = args.kernels
    kernel_size = args.kernel_size
    layers = args.layers
    model_name = f"model_{dims}_{kernels}k_{kernel_size}x{kernel_size}_{layers}l.pt"
    print("Model Name: ", model_name)

    # prepare to load the images
    print("Setting up the data loaders...")
    train_loader, train_dataset, test_loader, test_dataset = data.LoadImages(color=color, shuffle=False)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # train mode
    if args.mode == 'train':
        minecraft_model = model.MinecraftCNN(in_channels=3 if color else 1,
                                      num_kernels=kernels,
                                      kernel_size=kernel_size,
                                      num_conv_layers=layers)
        train.TrainModel(minecraft_model, train_loader)
        torch.save(minecraft_model, f"models/{model_name}")

    # evaluate mode
    elif args.mode == 'evaluate':
        minecraft_model = torch.load(f"models/{model_name}")
        evaluate.Evaluate(minecraft_model, test_loader)
        pass

    return


    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--color', action='store_true', default=True)
    parser.add_argument('--kernels', type=int, default=1)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    main()