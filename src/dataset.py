from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
from torch.utils.data import Dataset,DataLoader
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

class RECUPERATION_IMG(Dataset):
    def __init__(self, X, y,transform):
        self.img_path = X.values
        self.labels = torch.tensor(y.values, dtype=torch.long)
        if transform is None:
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),         
            transforms.ToTensor(),                
            transforms.Normalize(                   
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
            ]) 
        else:
            self.transform=transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self,idx):
        img_path=self.img_path[idx]
        label=self.labels[idx]
        img=Image.open(img_path).convert("RGB")
        img=self.transform(img)
        return img,label
try:
        chemin=r"C:\Users\AWOUNANG\Desktop\CLASSIFICATION ANIMAL\datas"
        folders =os.listdir(chemin)

        label_map={}
        with open("dataset.csv",mode="w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(["label_paths","targets"])
            for folder in folders:
                folder_path=os.path.join(chemin,folder)
                # print(folder_path)
                list=os.listdir(folder_path)
                for  lb in list:
                        label_path=os.path.join(folder_path,lb)  
                        label_map[lb]=folders.index(f"{folder}")
                        writer.writerow([label_path,label_map[lb]])
except Exception :
     pass



def loader_loop(csv_path,train_size,val_size):
       data=pd.read_csv(csv_path,sep=",")
       X=data["label_paths"]
       y=data["targets"]
       x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=train_size,shuffle=True)
       x_test,x_val,y_test,y_val=train_test_split(x_test,y_test,test_size=val_size,shuffle=True)
       train_dataset=RECUPERATION_IMG(x_train,y_train,transform=None)
       test_dataset=RECUPERATION_IMG(x_test,y_test,transform=None)
       val_dataset=RECUPERATION_IMG(x_val,y_val,transform=None)
       return train_dataset, test_dataset, val_dataset
device = torch.device("cpu")
batch_size=20
train_dataset, val_dataset, test_dataset =loader_loop(csv_path="dataset.csv",train_size=0.4,val_size=0.5)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loss, val_loss= [], []



