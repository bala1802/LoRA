import torch
import torch.nn as nn

class MNIST_MODEL(nn.Module):
    def __init__(self, size_of_hidden_layer_1 = 1000, size_of_hidden_layer_2 = 2000):
        super(MNIST_MODEL, self).__init__()

        self.linear_layer_1 = nn.Linear(28*28, size_of_hidden_layer_1)
        self.linear_layer_2 = nn.Linear(size_of_hidden_layer_1, size_of_hidden_layer_2)
        self.linear_layer_3 = nn.Linear(size_of_hidden_layer_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear_layer_1(x))
        x = self.relu(self.linear_layer_2(x))
        x = self.relu(self.linear_layer_3(x))
        return x