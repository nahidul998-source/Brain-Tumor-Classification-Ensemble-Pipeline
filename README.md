# Brain Tumor Classification Ensemble Pipeline

# Overview
This project implements a comprehensive deep learning pipeline for brain tumor classification using multiple CNN architectures with ensemble methods. The pipeline includes cross-validation, individual model training, weighted voting ensembles, stacking ensembles, and Grad-CAM visualization.

# Model Architectures
The pipeline supports six state-of-the-art CNN architectures:

MobileNetV2
MobileNetV3Large
ResNet50V2
EfficientNetB0
EfficientNetB4
VGG16

# Key Features
Data Pipeline
Image Augmentation: Random flips, brightness, contrast adjustments
Meningioma Oversampling: Targeted oversampling to address class imbalance
Label Smoothing: Reduces overfitting and improves generalization
TF.Data Pipeline: Optimized data loading with prefetching and parallel processing
Training Strategy

Two-Phase Training:
Phase 1: Head training with frozen backbone (8 epochs)
Phase 2: Fine-tuning with unfrozen last layers (10 epochs)
K-Fold Cross-Validation: 5-fold stratified cross-validation
Early Stopping: Prevents overfitting with patience of 5 epochs
Learning Rate Scheduling: Separate LRs for head (1e-3) and fine-tuning (1e-5)
Gradient Clipping: Prevents gradient explosion

Ensemble Methods
Weighted Soft Voting: Uses validation F1 scores as weights
Selective Top-K Ensemble: Automatically selects best-performing models
Stacking Ensemble: Meta-learner (Logistic Regression) on validation predictions
Visualization & Explainability

Grad-CAM++: Advanced visualization for model interpretability
Focused Heatmaps: Component-based heatmap focusing with connected components
Ensemble Grad-CAM: Combined visualization across multiple models
ROC Curves & Confusion Matrices: Comprehensive evaluation metrics

Performance Metrics
Accuracy, Macro-F1, Macro-Precision, Macro-Recall
IoU/Jaccard Score (macro)
AUC-ROC (macro)
Per-class metrics (precision, recall, specificity, F1)
Bootstrap confidence intervals (95% CI)
McNemar statistical tests for model comparison

# Usage
Configuration - python
Key configuration parameters:
MODELS_TO_RUN = ['mobilenetv2', 'mobilenetv3', 'resnet50v2', 
                 'efficientnetb0', 'efficientnetb4', 'vgg16']
N_SPLITS = 5
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 10
LR_HEAD = 1e-3
LR_FINE = 1e-5
DROPOUT_RATE = 0.4
DENSE_UNITS = 256
Running the Pipeline
python

# Full pipeline execution
python brain_tumor_classification.py
Output Structure
text
/kaggle/working/brain_tumor_results_advanced_bigv10.1/
├── models/              # Model weights (.h5)
├── figures/             # ROC curves, confusion matrices, training curves
├── predictions/         # Prediction probabilities and labels (.npy, .csv)
├── statistics/          # Performance metrics, per-class metrics, bootstrap CI
├── gradcam/             # Grad-CAM visualizations
├── checkpoints/         # Training checkpoints
├── logs/                # Training logs (CSV)
└── meta_learner/        # Stacking ensemble meta-learner models

# Results
The pipeline generates comprehensive evaluation results:
Individual Model Performance
Train/Validation/Test accuracy, F1, precision, recall, IoU
Per-class metrics for each model
Bootstrap confidence intervals
ROC curves and confusion matrices
Ensemble Performance
Weighted soft voting (validation F1 weights)
Selective top-K ensembles (K=2,3,4)
Logistic regression stacking
Statistical significance testing (McNemar)
Visualization Outputs
Training accuracy/loss curves
ROC curves for each model
Confusion matrices
Grad-CAM++ heatmaps (individual and ensemble)

# Requirements
GPU Requirements
GPU: NVIDIA P100 or compatible (Tensor Cores not required)
Framework: TensorFlow 2.x
Mixed Precision: Disabled (float32) for P100 compatibility

Python Dependencies
bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib
pip install opencv-python psutil joblib statsmodels kagglehub

Hardware Optimization
The pipeline includes specific optimizations for the NVIDIA P100 GPU:
Disabled mixed precision (float32 only)
Enabled XLA compiler for performance boost
Memory growth configuration
Automated TensorFlow environment optimization

Pipeline Workflow
Data Loading & Preprocessing
Downloads dataset via KaggleHub
Loads train/val/test splits
Generates class names and paths
Cross-Validation (Optional)
5-fold stratified CV on training data
Validates each model architecture

#Final Training
Two-phase training for each model
Saves best weights and predictions
Individual Evaluation
Comprehensive metrics computation
Visualization generation
Ensemble Construction
Weighted soft voting
Selective top-K ensembles
Stacking ensemble
Statistical Analysis
McNemar pairwise comparisons
Bootstrap confidence intervals
Visualization
Grad-CAM for individual models
Ensemble Grad-CAM

Key Performance Indicators
The pipeline tracks these primary metrics:
Test Accuracy: Overall classification accuracy
Macro F1-Score: Balanced measure across all classes
IoU/Jaccard Score: Intersection over Union
Macro AUC: Area under ROC curve (macro-averaged)

# Notes
Class Imbalance: The pipeline includes meningioma oversampling and label smoothing

Reproducibility: All random seeds are fixed (SEED=42)

Memory Management: Intensive memory cleanup and garbage collection

Resume Training: Supports resuming from checkpoints

Debug Mode: FAST_DEBUG flag for quick testing (1 epoch each phase)
