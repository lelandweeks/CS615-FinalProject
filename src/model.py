import torch.nn as nn

class MinecraftCNN(nn.Module):
    def __init__(self, in_channels=3, num_kernels=1, kernel_size=3, num_conv_layers=1, num_classes=60):
        super(MinecraftCNN, self).__init__()

        # hyperparameters for the convolutional and pooling layers
        CONV_STRIDE = 1
        POOL_KERNEL_SIZE = 2
        POOL_STRIDE = 2 

        # compute dimensions for the linear layer
        # image files input: 320x180 pixels, 3 color channels (RGB), no alpha channel
        conv_width  = (320 - kernel_size) / CONV_STRIDE + 1
        conv_height = (180 - kernel_size) / CONV_STRIDE + 1
        pool_width  = (conv_width  - POOL_KERNEL_SIZE) / POOL_STRIDE + 1
        pool_height = (conv_height - POOL_KERNEL_SIZE) / POOL_STRIDE + 1
        linear_input = int(num_kernels * pool_width * pool_height)

        # buildl the 'base' model
        self.model = nn.Sequential(
            # output width = (input_width - kernel_size + 2*padding) / stride + 1
            # output height = (input_height - kernel_size + 2*padding) / stride + 1
            # calculate width: (320 - 3 + 2*0) / 1 + 1 = 318
            # calculate height: (180 - 3 + 2*0) / 1 + 1 = 178
            nn.Conv2d(in_channels=in_channels, out_channels=num_kernels, kernel_size=kernel_size, stride=CONV_STRIDE, padding=0),
            
            # input is an image so we want all values to be non-negative
            # also, ReLU is default activation layer for CNNs
            nn.ReLU(),
            
            # reduce dimensions to most important features, reducing computational cost
            # output width = (input_width - kernel_size) / stride + 1
            # output height = (input_height - kernel_size) / stride + 1
            # calculate width: (318 - 2) / 2 + 1 = 159
            # calculate height: (178 - 2) / 2 + 1 = 89
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE), 
            
            # flatten the image into a vector so linear layer can determine the class scores for each image
            # 159x89 -> 1x14,191
            nn.Flatten(),

            # fully connected layer to output class scores
            # input size = num_kernels * height * width
            # calculation: 1 * 159 * 89 = 14,191
            nn.Linear(linear_input, num_classes) 
        )

    def forward(self, x):
        return self.model(x)