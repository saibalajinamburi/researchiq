---
title: ResearchIQ
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# ResearchIQ

ResearchIQ is an ONNX-powered scientific abstract classifier built for overlapping arXiv domains.

## What this Space does

- accepts a scientific abstract
- generates a quantized ONNX MiniLM embedding
- classifies it with an ONNX-exported model
- shows the top predicted categories with probabilities
- displays training and export metadata

## Current model

- final classifier: scaled Logistic Regression pipeline exported to ONNX
- classes: 15
- macro F1: 0.6655
- runtime artifact: `models/phase4_onnx/best_model.onnx`

## Included visuals

- class distribution
- abstract length distribution
- final model comparison table

## Notes

This Space runs in direct inference mode and does not require a separate FastAPI backend.
