# An Interpretable Hybrid Deep Learning and Interval Type-2 Fuzzy Framework for Breast Microcalcification Classification

This repository contains the implementation accompanying the manuscript:

**An Interpretable Hybrid Deep Learning and Interval Type-2 Fuzzy Framework for Breast Microcalcification Classification in Digital Mammography**

The proposed framework combines handcrafted texture descriptors, convolutional and transformer-based deep representations, attention-based feature fusion, Interval Type-2 Fuzzy inference, and Fuzzy ART classification for benign/malignant classification of breast microcalcifications in digital mammography.

---

## Framework Overview

The method consists of three complementary feature-extraction branches:

### 1. Multi-scale Wavelet Branch
- Daubechies-4 (`db4`) wavelet decomposition
- Multi-level texture and frequency-domain analysis
- Statistical wavelet descriptors
- PCA-based dimensionality reduction

### 2. EfficientNetV2 + CBAM Branch
- EfficientNetV2 backbone
- Convolutional Block Attention Module (CBAM)
- Local deep feature extraction
- 128-dimensional feature representation

### 3. Vision Transformer Branch
- ViT-B/16 architecture
- Global contextual feature extraction
- 256-dimensional feature representation

The feature representations are concatenated and refined using an attention-based fusion module. The fused representation is then processed by an Interval Type-2 Fuzzy inference system and classified using Fuzzy ART.

---

## Architecture

```text
Mammogram / ROI
       |
       +----------------+----------------+
       |                |                |
       v                v                v
   Wavelet-db4   EfficientNetV2+CBAM   ViT-B/16
       |                |                |
      20-D            128-D            256-D
       |                |                |
       +----------------+----------------+
                        |
                        v
               Feature Concatenation
                     404-D
                        |
                        v
                  MinMaxScaler
                        |
                        v
                  Linear Projection
                     256-D
                        |
                        v
            Multi-Head Self-Attention
                        |
                        v
                    LayerNorm
                        |
                        v
          Interval Type-2 Fuzzy Logic
                        |
                        v
                    Fuzzy ART
                        |
                        v
              Benign / Malignant
```

All preprocessing and learned transformations are fitted using training data only and are then applied unchanged to validation and test data.

---

## Repository Structure

```text
.
├── main.py
├── config.py
├── data.py
├── wavelet.py
├── model.py
├── fuzzy.py
├── art.py
├── train.py
├── requirements.txt
├── features/
├── results/
└── README.md
```

### Main Files

- `main.py`  
  Main command-line entry point for preprocessing, training, validation, testing, cross-dataset evaluation, sensitivity analysis, ablation experiments, and benchmarking.

- `config.py`  
  Dataset paths, experiment settings, random seeds, model parameters, and training configuration.

- `data.py`  
  Dataset loading, ROI handling, patient-level partitioning, and preprocessing.

- `wavelet.py`  
  Multi-scale wavelet decomposition, handcrafted feature extraction, and dimensionality-reduction utilities.

- `model.py`  
  EfficientNetV2 + CBAM, Vision Transformer, and attention-based feature-fusion components.

- `fuzzy.py`  
  Interval Type-2 Fuzzy membership functions, rule evaluation, type reduction, and fuzzy scoring.

- `art.py`  
  Fuzzy Adaptive Resonance Theory classifier.

- `train.py`  
  Training, validation, testing, metric calculation, and experiment utilities.

---

## Requirements

Recommended environment:

- Python 3.10 or 3.11
- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- PyWavelets
- OpenCV
- Pillow
- SciPy
- matplotlib
- SHAP

A CUDA-compatible GPU is recommended for deep feature extraction and model training.

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Datasets

The experiments use the following public mammography datasets:

- DDSM / CBIS-DDSM
- INbreast

The datasets are not distributed with this repository. They must be obtained from their official or authorized sources and used in accordance with their respective licenses and terms of use.

A typical dataset layout is:

```text
Datasets/
├── CBIS_DDSM/
│   ├── AllPng/
│   ├── Masks/
│   └── csv/
│       └── CBIS_DDSM.csv
│
└── INbreast/
    ├── AllPng/
    ├── Masks/
    └── csv/
        └── INbreast_table_noClusters.csv
```

Dataset paths and experiment settings can be configured in `config.py`.

---

## Usage

The main experiments are executed through:

```bash
python main.py <command>
```

To display the available options for a command:

```bash
python main.py <command> --help
```

### ROI Preparation

```bash
python main.py prepare-rois
```

Prepares mammographic regions of interest for subsequent processing.

### Train Feature Branches

```bash
python main.py train-branch
```

Trains or initializes the individual deep feature-extraction branches.

### Feature Extraction

```bash
python main.py extract
```

Extracts Wavelet, EfficientNetV2+CBAM, and Vision Transformer features.

### Cross-Validation

```bash
python main.py cv
```

Runs patient-level cross-validation.

Patient-level splitting is used to ensure that samples originating from the same patient do not appear in both training and evaluation partitions.

### Full Model Training

```bash
python main.py train
```

Trains the complete hybrid framework.

### Test Evaluation

```bash
python main.py test
```

Evaluates the trained model on the selected test partition.

### Independent Holdout Evaluation

```bash
python main.py holdout
```

Evaluates the model on the independent patient-level holdout set.

### External Validation

```bash
python main.py external
```

Runs the configured external validation experiment.

### Cross-Dataset Validation

```bash
python main.py cross-dataset
```

Supports evaluation in both directions:

```text
DDSM / CBIS-DDSM -> INbreast
INbreast         -> DDSM / CBIS-DDSM
```

For cross-dataset evaluation, preprocessing parameters, normalization statistics, PCA transformations, and learned model parameters are derived from the training dataset only and are applied unchanged to the external test dataset.

### Sensitivity Analysis

```bash
python main.py sensitivity
```

Evaluates the effect of selected model and Fuzzy ART hyperparameters on classification performance.

### Ablation Study

```bash
python main.py ablation
```

Measures the contribution of the main framework components, including:

- Wavelet features
- EfficientNetV2 + CBAM
- Vision Transformer
- Attention-based fusion
- Interval Type-2 Fuzzy inference
- Fuzzy ART classification

### Karnik-Mendel Validation

```bash
python main.py validate-km
```

Validates the type-reduction implementation used by the Interval Type-2 Fuzzy inference module.

### Computational Benchmark

```bash
python main.py benchmark
```

Measures computational performance, including model execution and inference characteristics.

---

## Evaluation Protocol

The implementation supports:

- Patient-level stratified cross-validation
- DDSM / CBIS-DDSM evaluation
- INbreast evaluation
- Pooled dataset evaluation
- Independent holdout evaluation
- External validation
- Cross-dataset validation
- Sensitivity analysis
- Ablation studies
- Computational benchmarking

To prevent information leakage, patient-level splitting is performed before model evaluation. Data-dependent transformations are fitted exclusively on the training portion of each experiment.

---

## Evaluation Metrics

The evaluation pipeline supports:

- Accuracy
- Sensitivity
- Specificity
- Precision
- F1-score
- ROC-AUC
- False Positive Rate
- Negative Predictive Value
- Brier Score
- Calibration analysis

Confidence intervals can be estimated using bootstrap resampling.

---

## Interval Type-2 Fuzzy Inference

The Interval Type-2 Fuzzy component is used to represent uncertainty in the fused feature space.

The implementation includes:

- lower and upper membership functions
- interval-valued rule firing strengths
- rule aggregation
- type reduction
- fuzzy malignancy scoring

The fuzzy rule base is used to provide an uncertainty-aware intermediate representation before final classification.

---

## Fuzzy ART Classification

Fuzzy Adaptive Resonance Theory is used as the final adaptive classifier.

The implementation includes:

- fuzzy AND operation
- choice function
- match function
- vigilance criterion
- category creation
- category adaptation
- incremental learning

The main vigilance parameter used in the experimental configuration is:

```text
rho = 0.85
```

Other Fuzzy ART parameters can be adjusted through the project configuration.

---

## Explainability

The framework supports complementary interpretability methods:

- Grad-CAM++ for spatial visualization
- SHAP for feature-level attribution
- fuzzy-rule activation analysis
- fuzzy malignancy scoring

These mechanisms provide spatial, feature-level, and rule-level information for analysis of model predictions.

---

## Reproducibility

For reproducible experiments, the following items should be preserved:

- exact package versions
- random seeds
- configuration files
- patient-level train/validation/test splits
- extracted feature files
- out-of-fold predictions
- evaluation outputs
- model checkpoints, where permitted by dataset licenses

Package versions can be recorded with:

```bash
pip freeze > environment.txt
```

Raw mammography images should not be uploaded to the repository unless redistribution is explicitly permitted by the corresponding dataset license.

---

## Notes on Experimental Results

Reported numerical results should be associated with the exact dataset versions, patient-level partitions, preprocessing settings, model configuration, and trained checkpoints used in the corresponding experiment.



---

## Citation

If this implementation is used in academic work, please cite the associated manuscript:

**An Interpretable Hybrid Deep Learning and Interval Type-2 Fuzzy Framework for Breast Microcalcification Classification in Digital Mammography**

Complete bibliographic information and DOI can be added after publication.

---

## Code Availability

The repository includes the implementation required for:

- ROI preprocessing
- Wavelet feature extraction
- EfficientNetV2 + CBAM feature extraction
- Vision Transformer feature extraction
- attention-based feature fusion
- Interval Type-2 Fuzzy inference
- Fuzzy ART classification
- cross-validation
- holdout evaluation
- external validation
- cross-dataset validation
- sensitivity analysis
- ablation analysis
- computational benchmarking

---

## Contact

For questions related to the implementation or reproducibility of the experiments, please open an issue in this repository.
