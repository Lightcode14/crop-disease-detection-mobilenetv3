import os

# Path to your training data root folder containing folders: healthy, disease, pest
train_root =train_root = os.path.join("CROP DATASET", "train")

class_labels = []

# Loop over each main folder in train_root
for main_folder in os.listdir(train_root):
    main_folder_path = os.path.join(train_root, main_folder)
    if os.path.isdir(main_folder_path):
        # List subfolders inside each main folder (these are your actual disease classes)
        for subfolder in os.listdir(main_folder_path):
            subfolder_path = os.path.join(main_folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                class_labels.append(subfolder)  # Add the disease class

# Save to class_labels.txt
with open("class_labels.txt", "w") as f:
    for label in sorted(class_labels):
        f.write(label + "\n")

print("Class labels saved successfully.")
