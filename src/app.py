import streamlit as st
import easyocr
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pathlib import Path

# ─── Config ───────────────────────────────────────────────
MODEL_PATH = "Sharvari22/layoutlmv3-funsd"
LABEL_LIST = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

LABEL_COLORS = {
    "HEADER":   "#FF4B4B",  # red
    "QUESTION": "#1F77B4",  # blue
    "ANSWER":   "#2CA02C",  # green
    "OTHER":    "#AAAAAA"   # grey
}

# ─── Load model (cached so it only loads once) ────────────
@st.cache_resource
def load_model():
    processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return processor, model

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

# ─── Helper: normalize box to 0-1000 ──────────────────────
def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]

# ─── Helper: run OCR ──────────────────────────────────────
def run_ocr(reader, image):
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

# ─── Helper: run LayoutLMv3 ───────────────────────────────
def run_layoutlm(processor, model, image, words, boxes):
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

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    results = []
    for word, box, pred_id in zip(words, boxes, predictions[:len(words)]):
        label = ID2LABEL.get(pred_id, "O")
        entity = label.split("-")[-1] if "-" in label else "OTHER"
        results.append({"word": word, "box": box, "label": label, "entity": entity})

    return results

# ─── Helper: draw results on image ────────────────────────
def draw_results(image, results):
    draw = ImageDraw.Draw(image)
    for item in results:
        box = item["box"]
        entity = item["entity"]
        color = LABEL_COLORS.get(entity, "#AAAAAA")
        draw.rectangle(box, outline=color, width=2)
    return image

# ─── Main App ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Document Intelligence",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 Layout-Aware Document Intelligence")
    st.markdown("Upload a scanned document to extract and classify its fields using **LayoutLMv3**")

    # Sidebar legend
    st.sidebar.title("Legend")
    for entity, color in LABEL_COLORS.items():
        st.sidebar.markdown(
            f'<span style="color:{color}">■</span> **{entity}**',
            unsafe_allow_html=True
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** LayoutLMv3-base")
    st.sidebar.markdown("**Dataset:** FUNSD")
    st.sidebar.markdown("**F1 Score:** 0.82")

    # Load models
    with st.spinner("Loading models..."):
        processor, model = load_model()
        reader = load_ocr()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a document image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Document")
            st.image(image, use_column_width=True)

        with st.spinner("Running OCR..."):
            words, boxes = run_ocr(reader, image)

        with st.spinner("Running LayoutLMv3..."):
            results = run_layoutlm(processor, model, image, words, boxes)

        # Draw on image
        annotated_image = image.copy()
        annotated_image = draw_results(annotated_image, results)

        with col2:
            st.subheader("Detected Fields")
            st.image(annotated_image, use_column_width=True)

        # Results table
        st.subheader("Extracted Text by Entity Type")
        headers = [r for r in results if "HEADER" in r["label"]]
        questions = [r for r in results if "QUESTION" in r["label"]]
        answers = [r for r in results if "ANSWER" in r["label"]]

        col3, col4, col5 = st.columns(3)

        with col3:
            st.markdown("### 🔴 Headers")
            for r in headers:
                st.markdown(f"- {r['word']}")

        with col4:
            st.markdown("### 🔵 Questions")
            for r in questions:
                st.markdown(f"- {r['word']}")

        with col5:
            st.markdown("### 🟢 Answers")
            for r in answers:
                st.markdown(f"- {r['word']}")

        # JSON export
        st.subheader("📥 Export Results")
        st.download_button(
            label="Download JSON",
            data=json.dumps(results, indent=2),
            file_name="document_results.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()