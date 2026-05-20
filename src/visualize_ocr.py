import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# Load image
img_path = Path("data/dataset/training_data/images/0000971160.png")
image = Image.open(img_path)

# Load OCR output
with open("data/ocr_output.json", "r") as f:
    ocr_data = json.load(f)

# Filter low confidence detections
# Why 0.3: anything below is likely noise/garbage text
ocr_data = [item for item in ocr_data if item['confidence'] > 0.3]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 16))

# Left: original image
axes[0].imshow(image)
axes[0].set_title("Original Scanned Document", fontsize=14)
axes[0].axis("off")

# Right: OCR detections
axes[1].imshow(image)
axes[1].set_title(f"EasyOCR Detections ({len(ocr_data)} regions)", fontsize=14)

for item in ocr_data:
    box = item['box']
    text = item['text'][:15]  # truncate long text
    conf = item['confidence']

    x, y = box[0], box[1]
    width = box[2] - box[0]
    height = box[3] - box[1]

    # Color by confidence
    # Green = high confidence, Orange = medium, Red = low
    if conf > 0.7:
        color = 'green'
    elif conf > 0.5:
        color = 'orange'
    else:
        color = 'red'

    rect = patches.Rectangle(
        (x, y), width, height,
        linewidth=1.5,
        edgecolor=color,
        facecolor='none'
    )
    axes[1].add_patch(rect)
    axes[1].text(
        x, y - 3, text,
        fontsize=5, color=color,
        bbox=dict(facecolor='white', alpha=0.5, pad=1)
    )

axes[1].axis("off")

# Legend
from matplotlib.lines import Line2D
legend = [
    Line2D([0], [0], color='green', label='High confidence (>0.7)'),
    Line2D([0], [0], color='orange', label='Medium confidence (0.5-0.7)'),
    Line2D([0], [0], color='red', label='Low confidence (<0.5)'),
]
axes[1].legend(handles=legend, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig("data/ocr_visualization.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved to data/ocr_visualization.png")
print(f"High confidence regions: {sum(1 for i in ocr_data if i['confidence'] > 0.7)}")
print(f"Medium confidence regions: {sum(1 for i in ocr_data if 0.5 < i['confidence'] <= 0.7)}")
print(f"Low confidence regions: {sum(1 for i in ocr_data if i['confidence'] <= 0.5)}")