"""Download the Alzheimer MRI dataset from Hugging Face into ./Database.

Images are stored in one folder per class (NonDemented, VeryMildDemented,
MildDemented, ModerateDemented), which is the layout expected by
utils_simulation.create_dataset().
"""

import os

from datasets import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.getenv("DATASET_PATH", os.path.join(PROJECT_ROOT, "Database"))
LABELS = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]


def main():
    dataset = load_dataset("Falah/Alzheimer_MRI")

    for label in LABELS:
        os.makedirs(os.path.join(BASE_PATH, label), exist_ok=True)

    for idx, item in enumerate(dataset["train"]):
        label_name = LABELS[item["label"]]
        item["image"].save(os.path.join(BASE_PATH, label_name, f"image_{idx}.jpg"))

    print(f"Saved {idx + 1} images to {BASE_PATH}")


if __name__ == "__main__":
    main()
