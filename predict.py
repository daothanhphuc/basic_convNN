import torch
from torchvision import models, transforms
from PIL import Image

# Load the model (same as you fine-tuned)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # Update this number with your number of classes
checkpoint_path = "my_models/best.pt"  # Use the correct path to your model checkpoint
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)
model.eval()

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the new image
image_path = '/home/phucdz/Downloads/bear.jpeg'  # Provide your image path here
image = Image.open(image_path)
image = image.convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Move image to the device
image = image.to(device)

# Make prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted_class = torch.max(outputs, 1)

data_path = "/data/train"  # Update this to your actual data path
csv_file_path = os.path.join(data_path, "_classes.csv")

# Read the CSV file
labels_df = pd.read_csv(csv_file_path)

# Extract class names (assuming they are in columns 2 and beyond)
class_names = list(labels_df.columns[2:])
print(class_names)

# Get the predicted class name
predicted_class_name = class_names[predicted_class.item()]

# Display the prediction
print(f"The predicted class is: {predicted_class_name}")
