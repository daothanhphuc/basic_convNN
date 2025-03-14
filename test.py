import torch
from model import CNN
from dataset import AnimalDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import os

def test():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the path to the checkpoint.
    # Change this to "last.pt" if you want to test the most recent checkpoint.
    checkpoint_path = os.path.join("my_models", "best.pt")
    
    # Define the transform (must be same as during training)
    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create the test dataset.
    # Here, we are reusing the AnimalDataset by setting is_train to False.
    # Adjust this if you have a separate test split.
    test_dataset = AnimalDataset(path="data", is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Initialize the model and load the trained weights
    model = CNN(len(test_dataset.classes)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad(): # save memory
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    test()
