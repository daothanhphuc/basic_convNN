import pandas as pd 
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Compose
import os
import matplotlib.pyplot as plt
import cv2

list_lable = []

class AnimalDataset(Dataset):
    def __init__(self, path, is_train, transform=None):
        self.transform = transform
        if is_train:
            data_path = os.path.join(path, "train")
        else:
            data_path = os.path.join(path, "test")

        self.labels_df = pd.read_csv(os.path.join(data_path, "_classes.csv"))
        self.classes = list(self.labels_df.columns[2:]) 
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for _, row in self.labels_df.iterrows():
            image_name = row["filepath"]
            image_path = os.path.join(data_path, image_name)
            self.image_paths.append(image_path)

            # Find the class where value == 1
            class_label = self.classes[row[2:].values.argmax()]  # Get class name
            class_index = self.class_to_idx[class_label]  # Convert class name to numerical index
            self.labels.append(class_index)
            list_lable.append(class_label)
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply transform
        if self.transform:
            image = self.transform(image)
        labels = self.labels[index]
        return image, labels

if __name__ == "__main__":
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    dataset = AnimalDataset("data", is_train=True, transform= transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    # print(dataset.class_to_idx)
    for images, labels in dataloader:
        print(images.shape)
        print(labels.shape)
        break

    # print(len(dataset.classes))