import os 
import sys
chemin= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(chemin)
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from src.dataset import RECUPERATION_IMG
from src.model import get_model
from src.train import train_loop
from src.test import test_loop
from src.dataset import loader_loop
from torch.optim import  Adam
import yaml
DEVICE = torch.device("cpu")  # ou "cuda" si GPU
with open("config.yaml") as f:
    config = yaml.safe_load(f)


def main():
    print("Lancement du projet Classification Animale")
    loss_fn=CrossEntropyLoss()
    # Hyperparamètres
    NUM_CLASSES = 5
    model=get_model(config["num_class"])
    optimizer =Adam(model.parameters(),lr=0.001)
    BATCH_SIZE = 20
    train_dataset, val_dataset, test_dataset =loader_loop(config["csv_path"],train_size=0.6,val_size=0.5)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model, train_loss, train_acc, val_loss, val_acc = train_loop(model=model,train_loader=train_loader,val_loader=val_loader,loss_fn=loss_fn,optimizer=optimizer,device=DEVICE,epochs=config["epochs"])
    fig, axs = plt.subplots(1, 2, figsize=(11,6))    
    axs[0].plot(train_loss, label="Train Loss")
    axs[0].plot(val_loss, label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()


    axs[1].plot(train_acc, label="Train Accuracy")
    axs[1].plot(val_acc, label="Val Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    plt.show()

    loss_fn = CrossEntropyLoss()
    test_loop(model=model,test_loader=test_loader,loss_fn=loss_fn,device=DEVICE)

    print("Fin d'Exécution")

if __name__ == "__main__":
    main()
