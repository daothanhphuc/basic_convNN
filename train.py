import pandas as pd 
import numpy as np
from PIL import Image
from model import CNN
from dataset import AnimalDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomHorizontalFlip, RandomRotation, RandomResizedCrop
from tqdm.autonotebook import tqdm
import os
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def plot_confusion_matrix(writer,cm, class_names, epoch):
    """ 
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap= "Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max()/2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion Matrix', figure, global_step=epoch)


def train():
    num_epochs = 30
    BATCH = 64 
    LR = 0.001
    momentum = 0.9
    best_acc = -1
    log_path = "my_tensorboard"
    checkpoint_path = "my_models"
    # train_transform = Compose([
    #     RandomHorizontalFlip(),
    #     RandomRotation(10), # degrees
    #     ToTensor(),
    #     Resize((224, 224))
    # ])
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = AnimalDataset(path="data", is_train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)
    val_dataset = AnimalDataset(path="data", is_train=False, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=4, drop_last=True)
    # model = CNN(len(train_dataset.classes)).to(device)

    model = models.resnet18(pretrained=True)

    # Modify the last fully connected layer to match the number of classes in your dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)

    # Freeze all layers except for the last fully connected layer
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum)
    num_iters = len(train_loader)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        progress_bar = tqdm(train_loader, colour = 'red')
        total_losses = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_losses.append(loss.item())
            avr_loss = np.mean(total_losses)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}]".format(epoch+1, num_epochs, avr_loss))
            writer.add_scalar("Train/loss", avr_loss, global_step=epoch*num_iters + iter)
            #backward pass
            optimizer.zero_grad() # zero the parameter gradients to avoid accumulation
            loss.backward()
            optimizer.step()
            
        # Validation    
        model.eval()  # Set model to evaluation mode
        total_losses = []
        all_labels = []
        all_preds = []
        progress_bar = tqdm(val_loader, colour = 'green')
        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                #forward pass
                images = images.to(device)
                all_labels.extend(labels) # append the labels to the list 
                labels = labels.to(device)
                outputs = model(images) #shape [batch, num_classes]

                predictions = torch.argmax(outputs, dim=1) #shape [batch]
                all_preds.extend(predictions.tolist())

                loss = criterion(outputs, labels)
                total_losses.append(loss.item()) # loss.item() is the average loss of the batch
        
        avr_loss = np.mean(total_losses)
        writer.add_scalar("Loss/val", avr_loss, global_step=epoch)
        acc = accuracy_score(all_labels, all_preds)
        writer.add_scalar("Accuracy/val", acc, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_preds), train_dataset.classes, epoch)
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "last.pt".format()) )
        if acc > best_acc:
            best_acc = acc
            check_point = os.path.join(checkpoint_path, f"best.pt")
            torch.save(model.state_dict(), check_point)
            
if __name__ == "__main__":
    train()