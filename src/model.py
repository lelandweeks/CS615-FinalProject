import torch.nn as nn

class MinecraftCNN(nn.Module):
    def __init__(self, in_channels=3, num_kernels=1, kernel_size=3, num_conv_layers=1, num_classes=60):
        super(MinecraftCNN, self).__init__()

        # hyperparameters for the convolutional and pooling layers
        CONV_STRIDE = 1
        POOL_KERNEL_SIZE = 2
        POOL_STRIDE = 2 

        # image size: 320x180 pixels
        # 3 color channels (RGB) or 1 channel for grayscale
        # no alpha channel
        conv_width = 320
        conv_height = 180    

        # define the layers
        layers = []
        current_channels = in_channels
        for i in range(num_conv_layers):
            # output width = (input_width - kernel_size + 2*padding) / stride + 1
            # output height = (input_height - kernel_size + 2*padding) / stride + 1
            # (for 1 conv layer) calculate width: (320 - 3 + 2*0) / 1 + 1 = 318
            # (for 1 conv layer) calculate height: (180 - 3 + 2*0) / 1 + 1 = 178
            layers.append(nn.Conv2d(in_channels=current_channels, out_channels=num_kernels, kernel_size=kernel_size, stride=CONV_STRIDE, padding=0))
            
            # input is an image so we want all values to be non-negative
            # also, ReLU is default activation layer for CNNs
            layers.append(nn.ReLU())

            # update dimensions for next conv layer if there are multiple conv layers
            # need to use // instead of / to get an integer value for the dimensions
            conv_width = (conv_width - kernel_size) // CONV_STRIDE + 1
            conv_height = (conv_height - kernel_size) // CONV_STRIDE + 1
            current_channels = num_kernels

            # reduce dimensions to most important features, reducing computational cost
            # output width = (input_width - kernel_size) / stride + 1
            # output height = (input_height - kernel_size) / stride + 1
            # (for 1 conv layer) calculate width: (318 - 2) / 2 + 1 = 159
            # (for 1 conv layer) calculate height: (178 - 2) / 2 + 1 = 89
            layers.append(nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE))
            conv_height = (conv_height - POOL_KERNEL_SIZE) // POOL_STRIDE + 1
            conv_width = (conv_width - POOL_KERNEL_SIZE) // POOL_STRIDE + 1        

        # flatten the image into a vector so linear layer can determine the class scores for each image
        # (for 1 conv layer) calculation: 159x89 -> 1x14,191
        layers.append(nn.Flatten())

        # fully connected layer to output class scores
        # input size = num_kernels * height * width
        # (for 1 conv layer) calculation: 1 * 159 * 89 = 14,191
        linear_input = int(current_channels * conv_width * conv_height)
        layers.append(nn.Linear(linear_input, num_classes))

        # build the model
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)