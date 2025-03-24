import pandas as pd 
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchvision import datasets
import os
import matplotlib.pyplot as plt
import cv2
import pickle

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

class CIFAR10Dataset(Dataset):
    def __init__(self, path, is_train=True, transform=None):
        self.transform = transform
        self.is_train = is_train
        if self.is_train:
            self.data_files = [os.path.join(path, f"data_batch_{i}") for i in range(1, 6)]
        else:
            self.data_files = [os.path.join(path, "test_batch")]

        self.images = []
        self.labels = []
        for data_file in self.data_files:
            with open(data_file, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                self.images.append(batch['data'])
                self.labels.append(batch['labels'])
        # concatenate images and labels
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.classes = self.load_classes(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # reshape the image from a flat array of 3072 elements to 32x32x3
        image = image.reshape(3, 32, 32).transpose((1, 2, 0))  # Convert to HxWxC format
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        return image, label

    def load_classes(self, path):
        """Load the class names from the 'batches.meta' file."""
        with open(os.path.join(path, 'batches.meta'), 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        return meta['label_names']

if __name__ == "__main__":
    transform = Compose([
        ToTensor(),
        # Resize((224, 224))
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    # dataset = AnimalDataset("data", is_train=True, transform= transform)
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    # # print(dataset.class_to_idx)
    # for images, labels in dataloader:
    #     print(images.shape)
    #     print(labels.shape)
    #     break

    # # print(len(dataset.classes))
    train_dataset = CIFAR10Dataset("data/cifar10", is_train=True, transform=transform)
    test_dataset = CIFAR10Dataset("data/cifar10", is_train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)