import torch
from src.dataset import loader_loop
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

train_dataset, val_dataset, test_dataset =loader_loop(config["csv_path"],train_size=None,val_size=None)
def test_loop(model, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            test_acc += (predicted == labels).sum().item()

    test_loss /= len(test_dataset)
    test_acc /= len(test_dataset)

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc*100:.2f}%")

    return test_loss, test_acc
