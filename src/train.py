import torch
import yaml
import os
import matplotlib.pyplot as plt
config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)


def train_loop(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0

        for images, label in train_loader:
            image= images.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            label_pred= model(image)
            loss = loss_fn(label_pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (label_pred.argmax(1) == label).sum().item()

        train_loss.append(total_loss / len(train_loader))
        train_acc.append(100 * correct / len(train_loader.dataset))

        
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for images, label in val_loader:
                img = images.to(device)
                label=label.to(device)
                label_pred = model(img)
                v_loss += loss_fn(label_pred, label).item()
                v_correct += (label_pred.argmax(1) == label).sum().item()

        val_loss.append(v_loss / len(val_loader))
        val_acc.append(100 * v_correct / len(val_loader.dataset))

    
    return model,train_loss, train_acc, val_loss, val_acc
#model,train_loss,train_acc,val_loss,val_acc=train_loop(config["num_class"],config["epochs"])


