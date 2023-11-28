import torch
import torch.nn as nn

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x

class ResNet18FC(nn.Module):
    def __init__(self, num_classes=10):  # Adjusted for CIFAR-10
        super(ResNet18FC, self).__init__()

        # Define fully connected layers to replace convolutional layers
        self.fc1 = FullyConnectedLayer(3 * 32 * 32, 512)  # Adjust the output features
        self.fc2 = FullyConnectedLayer(512, 256)  # Adjust the output features
        self.fc3 = FullyConnectedLayer(256, 128)  # Adjust the output features
        self.fc4 = FullyConnectedLayer(128, num_classes)  # Adjust the output features

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)

        return x

def ResNet18():
    return ResNet18FC()