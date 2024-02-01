import torch
import torchvision.datasets as datasets
import transform
import config

def load_mnist_dataset(train):
    return datasets.MNIST(root='./data', train=train, download=True, transform=transform.data_transformation())

def data_loader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)