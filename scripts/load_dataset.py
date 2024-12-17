from datasets import load_dataset
import os

# Load the dataset
dataset = load_dataset("Falah/Alzheimer_MRI")

# Define the base path for your project
base_path = r"C:\Users\Admin\Desktop\RT4\FederatedLearningBlockchain\Database"

# Define the labels and ensure directories exist
labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
for label in labels:
    os.makedirs(os.path.join(base_path, label), exist_ok=True)

# Save images to their respective folders
for idx, item in enumerate(dataset['train']):
    label = item['label']  # Label index
    image = item['image']  # Image data (PIL Image object)
    label_name = labels[label]  # Map label index to label name
    # Save the image using a unique name (index-based)
    image.save(os.path.join(base_path, label_name, f"image_{idx}.jpg"))
    print(f"Saved: {label_name}/image_{idx}.jpg")

print("Dataset downloaded and organized successfully!")