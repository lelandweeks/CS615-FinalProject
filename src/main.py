import data
import evaluate
import model
import train

def main():
    train_loader, train_dataset, test_loader, test_dataset = data.LoadData()


    # in_channels=3, num_kernels=1, kernel_size=3, num_conv_layers=1, num_classes=60):
    # model = model.MinecraftCNN()

    # 1 channel, 1 kernel, 3x3 kernel size, 1 conv layer
    model_1c_1k_3x3_1l = model.MinecraftCNN(in_channels=1, num_kernels=1, kernel_size=3, num_conv_layers=1)
    train = train.TrainModel(model_1c_1k_3x3_1l, train_loader)
    train.train()

    


if __name__ == "__main__":
    main()