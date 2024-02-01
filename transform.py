import torchvision.transforms as transforms

def data_transformation():
    return transforms.compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ),
                             (0.3081, ))
    ])