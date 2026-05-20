import json
import os
from pathlib import Path

# Load one annotation file
data_path = Path("data/dataset/training_data/annotations")
sample_file = list(data_path.glob("*.json"))[0]

with open(sample_file, "r") as f:
    data = json.load(f)

print(f"File: {sample_file.name}")
print(f"Number of form fields: {len(data['form'])}")
print("\nFirst field sample:")
print(json.dumps(data['form'][0], indent=2))