# Chest X-ray Finding Analyzer — Research Presentation Deck & Demo Script

Academic & Research Presentation Deck Outline (15 Slides) and Live Demonstration Protocol for the **Chest X-ray Finding Analyzer**.

---

## 📽️ 15-Slide Presentation Deck Outline

### Slide 1: Title & Project Overview
- **Title**: Chest X-ray Finding Analyzer: Zero-Shot Medical Vision-Language Analysis & Visual Attribution
- **Subtitle**: Powered by PubMedCLIP & FastAPI
- **Presenter**: Advanced Agentic AI Coding Research Team
- **Key Highlight**: Educational and research prototype for automated chest radiography feature extraction without task-specific retraining.

### Slide 2: Problem Statement & Motivation
- **Clinical Challenge**: Radiographic interpretation requires extensive specialized expertise; diagnostic backlogs impact global triage efficiency.
- **AI Gap**: Traditional supervised CNNs require huge annotated datasets and fail to generalize across unseen findings or novel prompts.
- **Solution**: Vision-language foundation models (PubMedCLIP) trained on biomedical literature enable zero-shot finding candidate scoring.

### Slide 3: PubMedCLIP Foundation Model
- **Model Card**: `flaviagiammarino/pubmed-clip-vit-base-patch32`
- **Architecture**: ViT-B/32 Vision Encoder paired with a Transformer Text Encoder fine-tuned on PubMed biomedical image-text pairs.
- **Representation**: Maps both $224 \times 224$ X-ray images and clinical text descriptions into a shared $512$-dimensional $L_2$-normalized vector space.

### Slide 4: End-to-End System Architecture
- **Pipeline Stages**: Client Upload $\rightarrow$ Image Preprocessing $\rightarrow$ PubMedCLIP Joint Embedding $\rightarrow$ Cosine Similarity Engine $\rightarrow$ Candidate Ranking $\rightarrow$ Spatial Patch Visual Attribution.
- **Tech Stack**: Python, FastAPI, PyTorch, Hugging Face Transformers, Pillow, OpenCV, HTML5/CSS3 Dark UI.

### Slide 5: Controlled Medical Prompt Set
- **Prompt Engineering**: Formulated 8 standardized clinical prompts targeting key findings:
  1. `Normal`: "A chest X-ray showing a normal healthy lung"
  2. `Pneumonia`: "A chest X-ray showing pneumonia"
  3. `Pleural Effusion`: "A chest X-ray showing pleural effusion"
  4. `Pneumothorax`: "A chest X-ray showing pneumothorax"
  5. `Cardiomegaly`: "A chest X-ray showing cardiomegaly"
  6. `Atelectasis`: "A chest X-ray showing atelectasis"
  7. `Pulmonary Edema`: "A chest X-ray showing pulmonary edema"
  8. `Consolidation`: "A chest X-ray showing consolidation"

### Slide 6: Preprocessing & Input Validation
- **Image Support**: PNG, JPEG, TIFF, BMP format validation.
- **Normalization**: Standardized RGB conversion, bilinear resizing to $224 \times 224$, float scaling to $[0, 1]$, and ImageNet normalization ($\mu=[0.481, 0.457, 0.408]$, $\sigma=[0.268, 0.261, 0.275]$).

### Slide 7: Similarity & Prototype Confidence Ranking
- **Cosine Dot-Product**: $S_i = \mathbf{e}_{\text{img}} \cdot \mathbf{e}_{\text{text}, i} \in [-1, 1]$
- **Rank Ordering**: Descending sort of similarity scores.
- **Prototype Confidence**: Maximum similarity score across candidate finding prompts.
- **Status Classification**: `Candidate` (score $\ge$ threshold) vs `Unlikely` (score $<$ threshold).

### Slide 8: Visual Spatial Attribution (Explainability)
- **Methodology**: Extracted $49$ spatial patch embeddings ($7 \times 7$ grid) from ViT-B/32 `last_hidden_state`.
- **Spatial Cosine Similarity**: Projected patch tokens to $512$-dim embedding space and computed dot-product against target text prompt embedding.
- **Bilinear Upsampling**: Resized $7 \times 7$ similarity map to original image resolution with warm colormap overlay (Red/Yellow = high similarity).

### Slide 9: Evaluation Dataset & Ground Truth Pipeline
- **Dataset Structure**: 16 sample chest X-rays annotated across 8 finding classes.
- **Split Management**: Partitioned into Validation (`val`, $N=8$) and Test (`test`, $N=8$) splits to prevent data leakage during calibration.

### Slide 10: Zero-Shot Quantitative Evaluation
- **Metrics Computed**: Precision, Recall / Sensitivity, Specificity, F1-Score, ROC-AUC, PR-AUC, Macro-Average, and Micro-Average.
- **Confusion Matrices**: Generated individual binary confusion matrix PNG plots for all 8 findings.

### Slide 11: Decision Threshold Calibration
- **Threshold Bias Challenge**: Uncalibrated zero-shot models often cluster dot-products near baseline values ($0.20 - 0.28$).
- **Grid Sweep**: Evaluated candidate grid $[0.15, 0.40]$ on Validation split.
- **Results**: Per-Finding dynamic calibration improved held-out Test F1-score from $0.0000$ (default $0.25$) to **$0.3843$**.

### Slide 12: Modern Web Application & Frontend UI
- **Design System**: Dark-mode glassmorphic interface with reactive micro-animations.
- **Interactive Features**: Drag-and-drop file uploader, image preview container, ranked findings table with status badges, and interactive click-to-view heatmap display.

### Slide 13: Containerized Deployment Architecture
- **Dockerization**: Multi-stage build based on `python:3.11-slim` with OpenCV headless bindings and non-root security user (`appuser`).
- **Orchestration**: `docker-compose.yml` service with persistent HuggingFace volume caching and `/health` monitoring.
- **Process Manager**: Production Gunicorn configuration with `UvicornWorker` processes.

### Slide 14: Medical Safety & Regulatory Disclaimers
- **Regulatory Status**: Research and educational prototype only. Not approved by FDA or CE for clinical diagnosis.
- **Terminology Enforcement**: Scores represent vector similarity, **NOT** diagnostic probabilities. Findings marked as "prototype candidate scores".

### Slide 15: Conclusion & Future Research Directions
- **Summary**: Delivered end-to-end zero-shot medical vision-language platform with visual explainability, threshold calibration, and production deployment.
- **Future Directions**: Fine-tuning on larger MIMIC-CXR dataset, multi-modal report generation, and integrated LLM clinical context summarization.

---

## 🎬 Recommended Live Demonstration Protocol

### Step 1: Launch Local Server & Frontend
```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```
Open browser to `http://127.0.0.1:8000/static/index.html`.

### Step 2: Upload Sample Chest X-Ray
1. Drag and drop `data/evaluation/images/sample_001.png` onto the upload dropzone.
2. Observe immediate image preview and file metadata validation.

### Step 3: Run Analysis
1. Click **"Analyze Chest X-Ray"**.
2. Observe loading spinner while PubMedCLIP generates image and prompt embeddings.
3. Review ranked findings table displaying scores, rank numbers, and `Candidate`/`Unlikely` badges.

### Step 4: Inspect Visual Explainability Heatmap
1. Note the automatically rendered spatial attribution heatmap overlay for the top-ranked finding in the right-hand panel.
2. Click on a different finding row (e.g., `Pneumonia` or `Cardiomegaly`) to dynamically request and display its corresponding $7 \times 7$ spatial patch overlay.

### Step 5: Verify Automated Test Suite
Run full verification suite in terminal:
```powershell
.venv\Scripts\python.exe -m pytest
```
Confirm 38/38 passing tests demonstrating complete system integrity.
