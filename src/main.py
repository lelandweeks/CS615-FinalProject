import argparse
import torch
import matplotlib.pyplot as plt

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
    COMMAND LINE: python main.py --mode train

2D MODEL: Same as base with grayscale images
    PARAMETERS: in_channels=1, num_kernels=1, kernel_size=3, num_conv_layers=1
    FILE NAME: model_gray_1k_3x3_1l.pt
    COMMAND LINE: python main.py --mode train --color False
 
MULTIPLE KERNELS MODEL: Same as base with multiple kernels
    PARAMETERS: in_channels=3, num_kernels=3, kernel_size=3, num_conv_layers=1
    FILE NAME: model_color_3k_3x3_1l.pt
    COMMAND LINE: python main.py --mode train --kernels 3
 
KERNEL SIZE MODEL: Same as base with 5x5 kernels
    PARAMETERS: in_channels=3, num_kernels=1, kernel_size=5, num_conv_layers=1
    FILE NAME: model_color_1k_5x5_1l.pt
    COMMAND LINE: python main.py --mode train --kernel_size 5

MULTIPLE CONV LAYERS MODEL: Same as base with multiple convolutional layers
    PARAMETERS: in_channels=3, num_kernels=1, kernel_size=3, num_conv_layers=3
    FILE NAME: model_color_1k_3x3_3l.pt
    COMMAND LINE: python main.py --mode train --layers 3
"""

def main():

    # parse the command line arguments to target the corresponding model
    args = parse_args()
    print(f"Mode: {args.mode}, Color: {args.color}, Kernels: {args.kernels}, Layers: {args.layers}, Kernel Size: {args.kernel_size}")

    color = True
    dims = 'color'
    if args.color == False:
        color
        dims = 'gray'
    kernels = args.kernels
    kernel_size = args.kernel_size
    layers = args.layers
    epochs = args.epochs
    learning_rate = args.lr
    base_filename = f"{dims}_{kernels}k_{kernel_size}x{kernel_size}_{layers}l"
    model_filename = "model_" + base_filename + ".pt"
    plot_filename = "plot_" + base_filename + ".png"
    print("Model Filename: ", model_filename)
    print("Plot Filename: ", plot_filename)

    # prepare to load the images
    print("Setting up the data loaders...")
    train_loader, train_dataset, test_loader, test_dataset = data.LoadImages(color=color, shuffle=False)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # train mode
    if args.mode == 'train':
        print(f"Training with pareamters: Epochs: {epochs}, Learning Rate: {learning_rate}")
        minecraft_model = model.MinecraftCNN(in_channels=3 if color else 1,
                                      num_kernels=kernels,
                                      kernel_size=kernel_size,
                                      num_conv_layers=layers)
        train_model = train.TrainModel(minecraft_model, train_loader)
        train_losses = train_model.train(num_epochs=epochs, learning_rate=learning_rate)
        torch.save({'model': minecraft_model.state_dict(), 'losses': train_losses}, f"models/{model_filename}")

        # save the plot
        plt.plot(train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Epochs')
        plt.savefig(f'plots/{plot_filename}')

    # evaluate mode
    elif args.mode == 'evaluate':
        minecraft_model = model.MinecraftCNN(in_channels=3 if color else 1,
                                      num_kernels=kernels,
                                      kernel_size=kernel_size,
                                      num_conv_layers=layers)
        model_from_disk = torch.load(f"models/{model_filename}", weights_only=False)
        minecraft_model.load_state_dict(model_from_disk['model'])
        train_losses = model_from_disk['losses']
        evaluate_model = evaluate.EvaluateModel(minecraft_model, test_loader)
        accuracy = evaluate_model.evaluate()
        print(f'Final Test Accuracy: {accuracy:.2%}')
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--color', action='store_true', default=True)
    parser.add_argument('--kernels', type=int, default=1)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()

if __name__ == "__main__":
    main()