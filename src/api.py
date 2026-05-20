from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import easyocr
import torch
import numpy as np
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pathlib import Path
import json
import io

# ─── Setup ────────────────────────────────────────────────
app = FastAPI(
    title="Document Intelligence API",
    description="Layout-aware document understanding using LayoutLMv3",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = "models/layoutlmv3-finetuned"
LABEL_LIST = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION',
              'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

# ─── Load models at startup ───────────────────────────────
@app.on_event("startup")
async def load_models():
    global processor, model, reader
    print("Loading LayoutLMv3...")
    processor = LayoutLMv3Processor.from_pretrained(
        MODEL_PATH, apply_ocr=False
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Loading EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    print("All models loaded!")

# ─── Helper functions ─────────────────────────────────────
def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]

def run_ocr(image):
    result = reader.readtext(np.array(image))
    words, boxes = [], []
    seen = set()
    for detection in result:
        box_pts, text, conf = detection
        if conf < 0.3:
            continue
        x_coords = [p[0] for p in box_pts]
        y_coords = [p[1] for p in box_pts]
        box = [int(min(x_coords)), int(min(y_coords)),
               int(max(x_coords)), int(max(y_coords))]
        key = (text, tuple(box))
        if key not in seen:
            seen.add(key)
            words.append(text)
            boxes.append(box)
    return words, boxes

def run_layoutlm(image, words, boxes):
    width, height = image.size
    norm_boxes = [normalize_box(b, width, height) for b in boxes]

    encoding = processor(
        image.convert("RGB"),
        words,
        boxes=norm_boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()

    results = []
    for word, box, pred_id in zip(words, boxes, predictions[:len(words)]):
        label = ID2LABEL.get(pred_id, "O")
        entity = label.split("-")[-1] if "-" in label else "OTHER"
        results.append({
            "word": word,
            "box": box,
            "label": label,
            "entity": entity
        })
    return results

# ─── Routes ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "Document Intelligence API",
        "version": "1.0.0",
        "endpoints": ["/extract", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "layoutlmv3-base-finetuned"}

@app.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    """
    Upload a document image and get structured extraction results.
    Returns words, bounding boxes, and entity labels.
    """
    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images are supported"
        )

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run pipeline
    words, boxes = run_ocr(image)

    if not words:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the document"
        )

    results = run_layoutlm(image, words, boxes)

    # Organize by entity type
    headers = [r for r in results if "HEADER" in r["label"]]
    questions = [r for r in results if "QUESTION" in r["label"]]
    answers = [r for r in results if "ANSWER" in r["label"]]

    return {
        "filename": file.filename,
        "total_regions": len(results),
        "entities": {
            "headers": headers,
            "questions": questions,
            "answers": answers
        },
        "raw_results": results
    }