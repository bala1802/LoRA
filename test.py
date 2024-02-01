import torch
from tqdm import tqdm

import config

def test(model, data_loader):
    correct_predictions_count = 0
    wrong_predictions_count = [0 for i in range(10)]

    total = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Testing"):
            x, y = data
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            output = model(x.view(-1, config.INPUT_DIMENSION))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct_predictions_count += 1
                else:
                    wrong_predictions_count[y[idx]] += 1
                total += 1
        
    print(f"Accuracy : {round(correct_predictions_count/total, 2)}")
    for i in range(len(wrong_predictions_count)):
        print(f"Wrong Counts for the digit {i} : {wrong_predictions_count[i]}")

print("executed")




