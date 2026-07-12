# Sickle Cell Disease Detection Pipeline & Clinical Dashboard

## Overview
This repository features a complete, automated deep learning workflow for identifying Sickle Cell Anaemia from blood smear microscopy images. Moving past basic accuracy metrics, our approach heavily emphasizes clinical safety, model interpretability, and optimal hyperparameter discovery. 

A core goal of this project is to benchmark a cutting-edge Vision Transformer (Swin-Base) against a traditional Convolutional Neural Network (ResNet50) to see which architecture better handles the complex, irregular shapes of malformed blood cells.

Also included an **interactive web dashboard** to visualize real-time predictions and heatmaps!

---

##  Interactive Streamlit Dashboard
A lightweight, user-friendly clinical dashboard built with Streamlit to easily interact with the trained models. 

### Features:
* **Live Inference:** Upload any microscopic blood smear image to instantly get a diagnostic prediction.
* **Model Toggle:** Seamlessly switch between the Swin Transformer and ResNet50 models.
* **Visual Auditing (Grad-CAM):** See exactly *where* the model is looking. The dashboard generates spatial heatmaps to prove the model is focusing on cellular deformities rather than background noise.
* **Adjustable Clinical Threshold:** A slider allows you to manually tweak the decision threshold to prioritize 95% Recall (minimizing false negatives).

### How to Run the Dashboard:
```bash
# 1. Create and activate a virtual environment (Windows example)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install the necessary dependencies
pip install -r requirements.txt

# 3. Launch the web application
streamlit run app.py
```

---

## 1. Project Motivation
When dealing with medical diagnostics—especially blood disorders like Sickle Cell Anaemia—a False Negative (missing a sick patient) can be fatal. Therefore, a diagnostic AI must do more than just score high on standard accuracy; it needs to operate within strict safety limits. This study explores whether the self-attention mechanisms of Vision Transformers offer a safer, more reliable alternative to the localized filters of standard CNNs, while enforcing rigorous audits to prevent the AI from "cheating" by memorizing irrelevant image artifacts.

## 2. Technical Approach

### Data Management
* **Sourcing:** The image data is automatically fetched from the Sickle Cell Anaemia dataset via Kaggle.
* **Handling Imbalance:** Rather than generating fake data to balance the classes (which can introduce dangerous noise), we use a dynamically computed `WeightedRandomSampler` to ensure rare classes are sampled more frequently during training.

### Intelligent Hyperparameter Search
Instead of guessing parameters, the pipeline uses **Optuna** for Bayesian optimization.
* We actively search for the best learning rates and weight decay settings.
* A `MedianPruner` stops bad trials early, saving massive amounts of compute time and allowing the system to run efficiently on GPUs with limited VRAM (e.g., 4GB).

### Hardware Efficiency
To make training large models like Swin-Base feasible on consumer hardware, the code heavily utilizes PyTorch Automatic Mixed Precision (AMP), gradient scaling, and aggressive memory clearing (`gc.collect()`).

## 3. Prioritizing Patient Safety (Recall Optimization)
Most out-of-the-box machine learning models split decisions down the middle at a `50%` probability. In healthcare, this is dangerous. 

To fix this, we engineered a custom Precision-Recall evaluator. This script analyzes the model's performance on unseen data and recalculates the exact probability boundary needed to guarantee a **minimum 95% Recall**. This aggressively cuts down False Negatives, ensuring maximum patient safety.

## 4. Model Showdown: Swin vs. ResNet

The system automatically tests both architectures against a perfectly stratified hold-out dataset.

| Architecture | Overall Accuracy | Positive F1-Score | Safe Decision Threshold (95% Recall) |
| :--- | :---: | :---: | :---: |
| **Swin Transformer (Base)** | **96.49%** | **0.9756** | **0.8950** |
| **ResNet50 (Baseline)** | 92.98% | 0.9524 | 0.2913 |

**Key Takeaways:**
The Swin Transformer significantly outperformed the CNN in recognizing complex cellular geometries. Even better, it was highly confident: it could raise its decision threshold all the way to `0.8950` and still catch 95% of positive cases without triggering massive false alarms. To reach that same 95% safety net, the ResNet50 had to drop its threshold to a highly uncertain `0.2913`.

## 5. Explainability (Grad-CAM)
A medical AI is useless if it's looking at the wrong things (like camera glare or slide edges). 
* We integrated **Grad-CAM** to generate heatmaps of the models' "thought process".
* We applied Eigen and Augmentation smoothing to clean up the visual noise.
* The resulting audits prove that our models are genuinely analyzing the sickle-shaped anomalies of the red blood cells, ensuring clinical trustworthiness.

## 6. Codebase Architecture

```text
sickle_cell_project/
├── app.py                  # Interactive Streamlit Web Dashboard
├── main.py                 # Core Orchestrator (HPO tuning & Model Showdown)
├── requirements.txt        # Python package dependencies
└── src/                    
    ├── config.py           # Global settings, paths, and search spaces
    ├── data_setup.py       # Augmentations, samplers, and dataloading
    ├── model.py            # Model factory (using timm)
    ├── engine.py           # Training and validation loops with AMP
    ├── explainability.py   # Smoothed Grad-CAM visualization generation
    └── utils.py            # Plotting and clinical threshold optimization
```