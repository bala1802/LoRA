import torch
import torch.nn as nn
from tqdm import tqdm

import model_utils
import config

def fine_tune(train_loader, model, epochs, total_iterations_limit):
    cross_entropy_loss = model_utils.initialize_loss()
    optimizer = model_utils.initialize_optimizer(model=model, learning_rate=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        model.train()

        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            optimizer.zero_grad()
            output = model(x.view(-1, config.INPUT_DIMENSION))
            loss = cross_entropy_loss(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations >= total_iterations_limit:
                return