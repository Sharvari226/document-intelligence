# Layout-Aware Document Intelligence System

A multimodal AI system that extracts, understands, and queries scanned documents using OCR + LayoutLMv3 + Transformers.

## What it does
- Extracts text and layout from scanned documents using PaddleOCR
- Understands document structure using LayoutLMv3 (multimodal transformer)
- Answers natural language questions about documents
- Interactive Streamlit dashboard for real-time demo

## Tech Stack
PaddleOCR | LayoutLMv3 | HuggingFace | PyTorch | Streamlit | FastAPI

## Dataset
[FUNSD](https://guillaumejaume.github.io/FUNSD/) - Form Understanding in Noisy Scanned Documents

## Progress
- [x] Dataset exploration
- [x] Bounding box visualization
- [x] OCR pipeline
- [x] OCR visualization with confidence scoring
- [x] OCR pipeline
- [x] OCR visualization with confidence scoring
- [x] LayoutLMv3 fine-tuning (F1: 0.82)
- [x] Streamlit dashboard
- [x] FastAPI backend

## Demo
![Document Intelligence Demo](assets/demo.png)
<img width="1920" height="1008" alt="Screenshot (305)" src="https://github.com/user-attachments/assets/9e700afc-a9a9-4de1-adf1-0236c9c9af8c" />


## Sample Visualization
*(coming soon)*
