import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# Paths
ann_path = Path("data/dataset/training_data/annotations/0000971160.json")
img_path = Path("data/dataset/training_data/images/0000971160.png")

# Load annotation
with open(ann_path, "r") as f:
    data = json.load(f)

# Load image
image = Image.open(img_path)

# Color map for labels
color_map = {
    "question": "blue",
    "answer": "green",
    "header": "red",
    "other": "orange"
}

# Plot
fig, ax = plt.subplots(1, figsize=(12, 16))
ax.imshow(image)

for field in data["form"]:
    box = field["box"]
    label = field["label"]
    text = field["text"][:20]  # truncate long text
    
    x, y = box[0], box[1]
    width = box[2] - box[0]
    height = box[3] - box[1]
    color = color_map.get(label, "purple")
    
    # Draw bounding box
    rect = patches.Rectangle(
        (x, y), width, height,
        linewidth=1.5,
        edgecolor=color,
        facecolor="none"
    )
    ax.add_patch(rect)
    
    # Add label text
    ax.text(x, y - 5, f"{label}: {text}",
            fontsize=6, color=color,
            bbox=dict(facecolor="white", alpha=0.5, pad=1))

# Legend
from matplotlib.lines import Line2D
legend = [
    Line2D([0], [0], color="blue", label="Question"),
    Line2D([0], [0], color="green", label="Answer"),
    Line2D([0], [0], color="red", label="Header"),
    Line2D([0], [0], color="orange", label="Other"),
]
ax.legend(handles=legend, loc="upper right", fontsize=10)
ax.set_title("FUNSD Document - Bounding Box Visualization")
plt.tight_layout()
plt.savefig("data/sample_visualization.png", dpi=150)
plt.show()
print("Saved to data/sample_visualization.png")