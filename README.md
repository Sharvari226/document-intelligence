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

# Layout-Aware Document Intelligence System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-url.streamlit.app)

> Fine-tuned LayoutLMv3 on FUNSD achieving **0.82 F1 score** for document understanding

## 🔗 Live Demo
- **Streamlit Dashboard:** : https://sharvari-document-intelligence.streamlit.app/
- **HuggingFace Model:** [Sharvari22/layoutlmv3-funsd](https://huggingface.co/Sharvari22/layoutlmv3-funsd)
- **GitHub:** [Sharvari226/document-intelligence](https://github.com/Sharvari226/document-intelligence)
