import torch
import torchvision.datasets as datasets
import transform

def load_mnist_dataset(train):
    return datasets.MNIST(root='./data', train=train, download=True, transform=transform.data_transformation())

def data_loader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)