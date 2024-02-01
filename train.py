import torch
import torch.nn as nn
from tqdm import tqdm

import model_utils as model_utils
import config

def train(data_loader, model, epochs):
    cross_entropy_loss = model_utils.initialize_loss()
    optimizer = model_utils.initialize_optimizer(model=model, learning_rate=0.001)

    for epoch in range(epochs):
        model.train()
    
        loss_sum = 0
        num_of_iterations = 0
        data_iterator = tqdm(data_loader, desc=f'Epoch {epoch+1}')

        for data in data_iterator:
            x, y = data
            x = x.to(config.device)
            y = y.to(config.device)

            optimizer.zero_grad()
            output = model(x.view(-1, config.INPUT_DIMENSION))
            loss = cross_entropy_loss(output, y)
            loss_sum += loss.item()
            num_of_iterations += 1

            average_loss = loss_sum/num_of_iterations

            data_iterator.set_postfix(loss=average_loss)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    print("executed")