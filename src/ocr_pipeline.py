import easyocr
import json
from pathlib import Path

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Run on sample image
img_path = "data/dataset/training_data/images/0000971160.png"
result = reader.readtext(img_path)

# Parse results - deduplicate
extracted = []
seen = set()

for detection in result:
    box_points = detection[0]
    text = detection[1]
    confidence = detection[2]

    x_coords = [pt[0] for pt in box_points]
    y_coords = [pt[1] for pt in box_points]
    simple_box = [
        int(min(x_coords)),
        int(min(y_coords)),
        int(max(x_coords)),
        int(max(y_coords))
    ]

    # Deduplicate based on text + box position
    key = (text, tuple(simple_box))
    if key not in seen:
        seen.add(key)
        extracted.append({
            "text": text,
            "box": simple_box,
            "confidence": round(confidence, 3)
        })

# Save output
output_path = Path("data/ocr_output.json")
with open(output_path, "w") as f:
    json.dump(extracted, f, indent=2)

print(f"Extracted {len(extracted)} unique text regions")
print("\nSample output:")
for item in extracted[:5]:
    print(f"  Text: '{item['text']}' | Box: {item['box']} | Confidence: {item['confidence']}")