import os
import shutil
import random
import pandas as pd

# Set paths
source_folder = "data/animals"  # Contains subfolders for each species
train_folder = "data/train" 
test_folder = "data/test"

# Clear destination directories if they exist; otherwise, create them
for folder in [train_folder, test_folder]:
    if os.path.exists(folder):
        print(f"Deleting existing folder: {folder}")
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Get list of species (subfolders) and sort alphabetically
species = sorted([d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))])
print(f"Species folders found: {species}")

# Create subfolders for each class in both train and test directories
for sp in species:
    os.makedirs(os.path.join(train_folder, sp), exist_ok=True)
    os.makedirs(os.path.join(test_folder, sp), exist_ok=True)

train_ratio = 0.8
train_rows = []
test_rows = []

# For each species, shuffle images and split them
for sp in species:
    sp_source_folder = os.path.join(source_folder, sp)
    # List all images (filter by common image extensions)
    images = [img for img in os.listdir(sp_source_folder) if img.lower().endswith(('.jpg'))]
    print(f"Processing '{sp}': found {len(images)} images.")
    
    random.shuffle(images)
    n_train = int(len(images) * train_ratio)
    train_images = images[:n_train]
    test_images = images[n_train:]
    
    # Process training images
    for img in train_images:
        src_path = os.path.join(sp_source_folder, img)
        # Destination: data/train/species/img
        dest_path = os.path.join(train_folder, sp, img)
        shutil.copy(src_path, dest_path)
        print(f"Copied to train: {src_path} -> {dest_path}")
        
        # Prepare CSV row; filepath relative to train folder
        row = {"filepath": os.path.join(sp, img), "class": sp}
        for s in species:
            row[s] = 1 if s == sp else 0
        train_rows.append(row)
    
    # Process test images
    for img in test_images:
        src_path = os.path.join(sp_source_folder, img)
        dest_path = os.path.join(test_folder, sp, img)
        shutil.copy(src_path, dest_path)
        print(f"Copied to test: {src_path} -> {dest_path}")
        
        row = {"filepath": os.path.join(sp, img), "class": sp}
        for s in species:
            row[s] = 1 if s == sp else 0
        test_rows.append(row)

# Create DataFrames for train and test CSV files
columns = ["filepath", "class"] + species  # CSV header: filepath, class, then one-hot encoded species
train_df = pd.DataFrame(train_rows, columns=columns)
test_df = pd.DataFrame(test_rows, columns=columns)

# Save the CSV files in the corresponding train and test folders
train_csv_path = os.path.join(train_folder, "_classes.csv")
test_csv_path = os.path.join(test_folder, "_classes.csv")
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print("Data splitting complete!")
print(f"Total train images: {len(train_rows)}")
print(f"Total test images: {len(test_rows)}")
