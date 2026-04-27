# Multi-Modal Biometric System: Fingerprint & Iris Recognition

## Project Overview

This project implements a **multi-modal biometric recognition system** combining **fingerprint** and **iris** modalities. The system performs feature extraction, matching, and evaluation for each modality independently, with fusion currently under development. The fingerprint module uses minutiae-based triangle features with Delaunay triangulation, while the iris module uses Graph Spectral Normalised Embedding (GSNE) on normalised iris images.

---

## Table of Contents

1. [Fingerprint Recognition Module](#fingerprint-recognition-module)
2. [Iris Recognition Module](#iris-recognition-module)
3. [Fusion Module (In Progress)](#fusion-module-in-progress)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Acknowledgments](#acknowledgments)

---

## Fingerprint Recognition Module

### Overview

The fingerprint pipeline follows a minutiae-based approach enhanced with triangle-based geometric features, orientation fields, and a machine learning classifier. The main steps are:

1. Preprocessing & Quality Assessment
2. Minutiae Extraction (from skeleton or pre-computed `.xyt` files)
3. Delaunay Triangulation on minutiae points
4. Triangle Feature Extraction (geometric, orientation, ridge, structural)
5. Matching – three strategies combined:
   - Feature-based matching (Hungarian algorithm on triangle features)
   - Spatial overlap (KD-Tree)
   - ML classifier (Random Forest)
6. Score Generation & Evaluation (ROC, EER, d-prime, identification rate)

### Detailed Pipeline

#### 1. Configuration

The `FingerprintConfig` class centralises all hyperparameters:
- Preprocessing: CLAHE parameters, Gaussian blur kernel, morphological kernel size
- Minutiae extraction: threshold, maximum minutiae count, block size
- Feature extraction: window sizes, neighbourhood radius, circle radius
- Matching: feature weights, distance decay, max comparisons
- ML parameters: Random Forest with 50 trees, max depth 20
- Quality threshold: 0.2 (skip poor quality prints)

#### 2. Fingerprint Processor

The `FingerprintProcessor` class implements:

**Preprocessing (`preprocess_fingerprint`)**:
- Normalisation, CLAHE, Gaussian blur
- Otsu binarisation, morphological cleaning
- Thinning (skeletonisation) using Zhang-Suen algorithm
- Returns skeleton, enhanced image, binary image, and quality score

**Quality Assessment (`_calculate_ridge_quality_improved`)**:
- Ridge density: fraction of ridge pixels in skeleton
- Orientation consistency: based on gradient variance
- Minutiae quality: number of detected minutiae (target ~35)
- Ridge clarity: contrast in enhanced image
- Ridge continuity: measures breaks in skeleton

**Minutiae Extraction (`extract_minutiae`)**:
- Uses crossing number (CN) method on skeleton
- CN = 1 → ridge ending
- CN = 3 → bifurcation

**Orientation Field (`estimate_orientation_field`)**:
- Gradient-based method using Sobel operator
- Computes local ridge orientation in blocks

#### 3. Data Loading

The system automatically discovers fingerprint files from directories:
- Image directory (supports `.tif`, `.bmp`, `.png`, `.jpg`, `.jpeg`)
- Minutiae directory (`.xyt` files from FVC2000 databases)

Quality-based filtering:
- Skips images with size <10,000 or >1,000,000 pixels
- Skips prints with quality < 0.2
- Adaptive threshold for minutiae loading based on quality

#### 4. Feature Extraction

The `EnhancedFeatureExtractor` class transforms minutiae into triangle-based features:

**Delaunay Triangulation**: Creates triangles from minutiae points

**Geometric Features** (per triangle):
- Side ratios (a/b, b/c where a≤b≤c)
- Two smallest internal angles
- Normalised area = area / (largest side)²
- Inradius-to-circumradius ratio
- Aspect ratio and compactness
- Moment ratio from covariance of vertices

**Orientation Features**:
- Pairwise orientation differences (handling circular angles)
- Average, maximum, and standard deviation of differences
- Minutiae type pattern (sum and variance of types)

**Ridge Features**:
- Ridge frequency via FFT on horizontal projection
- Orientation smoothness (circular variance in 3×3 block)
- Orientation dispersion (circular variance in circular region)
- Ridge curvature (variance of orientations)

**Structural Features**:
- Area ratio with neighbouring triangles
- Shape consistency based on side ratios
- Adjacency degree (number of neighbouring triangles)

#### 5. Matching Strategies

**Feature-Based Matching** (`compare_features_enhanced`):
- Normalises triangle features (min-max)
- Assigns discriminativity weights based on variance
- Computes weighted distance matrix between triangles of two fingerprints
- Uses Hungarian algorithm for optimal assignment
- Converts average distance to similarity score

**Spatial Overlap** (`count_triangle_overlap_kdtree`):
- Builds KD-Trees of minutiae points
- Counts mutual nearest neighbours within threshold
- Also compares triangle areas of sampled triangles

**ML-Based Matching** (`train_enhanced_ml_matcher`):
- Random Forest classifier trained on fingerprint pairs
- Features: mean differences, std differences, triangle count ratio, feature-based similarity
- Balanced genuine and impostor pairs
- Hyperparameter tuning with GridSearchCV

**Final Score** (`enhanced_compare_final_improved`):
- Dynamic weights based on confidence (85% feature, 10% overlap, 5% ML for high confidence)
- Quality-based boosting
- Non-linear scaling for better separation

#### 6. Evaluation

**ROC Analysis** (`plot_enhanced_roc_analysis`):
- ROC curve with AUC
- Precision-Recall curve
- Optimal threshold (Youden's J statistic)
- Equal Error Rate (EER)

**Score Distribution** (`plot_enhanced_distributions`):
- Histograms of genuine vs impostor scores
- Box plots with statistics
- Cumulative Distribution Functions (CDF)

**Per-Person Identification** (`per_person_identification_enhanced`):
- For each person, compares all within-person fingerprint pairs
- Calculates identification rate (fraction ≥ threshold)
- Quality vs score scatter plots
- Summary CSV export

### Mathematical Formulations

#### Crossing Number (CN) for Minutiae Detection

```
CN = (1/2) Σ|pₖ - pₖ₊₁|, for k = 1 to 8
```

where pₖ are binary values of eight neighbours. CN=1 → ridge ending, CN=3 → bifurcation.

#### Triangle Geometric Features

For triangle with sides a ≤ b ≤ c:
- Side ratios: r₁ = a/b, r₂ = b/c
- Angles: α, β, γ (two smallest kept)
- Normalised area: Aₙₒᵣₘ = Area / c²
- Inradius-to-circumradius: r/R = (8A²)/(abc(a+b+c))

#### Weighted Similarity Score

```
S = w_f·S_fₑₐₜ + w_o·Sₒᵥₑᵣₗₐₚ + wₘ·Sₘₗ
```
where weights are dynamic based on confidence (0.85, 0.10, 0.05 for high confidence)

---

## Iris Recognition Module

### Overview

The iris module uses **Graph Spectral Normalised Embedding (GSNE)** to extract compact feature vectors from normalised iris images. The pipeline is:

1. Iris Localisation – pupil and iris boundary detection using Hough circles
2. Rubber Sheet Normalisation – mapping the iris annulus to a fixed-size rectangle (64×512)
3. Quality-based Pre-filtering – skip low-quality images (valid area <60%, texture std <0.08)
4. GSNE Feature Extraction – local graph construction on image patches, eigenvalue computation, aggregation into 10-dimensional feature vector
5. Similarity Calculation – weighted combination of cosine, Euclidean, and Manhattan similarities
6. Performance Evaluation – genuine/impostor scores, ROC, EER, d-prime

### Detailed Pipeline

#### 1. Dataset Configuration

- Uses **CASIA Iris Image Database (version 1.0)**
- Recursively finds all `.bmp` images
- Output folder for cached features (`.npy` files)

#### 2. Iris Localisation (`corrected_iris_localization`)

**Pupil Detection**:
- CLAHE enhancement (clip limit 3.0, 8×8 tiles)
- Hough Circle Transform (radius 15-45)
- Circle closest to image centre is selected

**Iris Detection**:
- Hough circles with radii [2×rₚ, 4×rₚ]
- Centre constrained within 30 pixels of pupil centre
- Multiple param2 values (50,45,40,35) for robustness

#### 3. Mask Creation (`create_better_mask`)

- Draws filled circle for iris outer boundary
- Removes pupil area (plus 5-pixel margin)
- Removes eyelid regions via two ellipses

#### 4. Rubber Sheet Normalisation (`diagnostic_rubber_sheet_transform`)

The annular iris region is mapped to a fixed-size rectangle:

```
Iₙₒᵣₘ(r,θ) = I(x_c + r·cosθ, y_c + r·sinθ)
```

- Radial resolution: 64
- Angular resolution: 512
- Inner radius: rₚ + 8
- Outer radius: rᵢ - 8
- Invalid pixels set to 0.5 (mid-grey)

#### 5. Graph Spectral Normalised Embedding (GSNE)

**Patch Division**:
- Patch sizes: (16,32) and (8,32)
- Patches extracted by sliding over 64×512 image

**Patch Quality Filtering**:
- Valid pixel ratio ≥ 0.7
- Texture standard deviation ≥ 0.08
- Contrast and Laplacian sharpness metrics

**Graph Construction** (per high-quality patch):
- 4-neighbour graph on valid pixels
- Edge weights: Wᵢⱼ = exp(-(Iᵢ - Iⱼ)²/σ²)
- σ = 70th percentile of intensity differences

**Spectral Features**:
- Compute normalised Laplacian: L = I - D^{-1/2} W D^{-1/2}
- Obtain eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λₘ
- Keep first 3 non-zero eigenvalues + mean + std → 5D per patch

**Aggregation**:
- For each patch size, collect all 5D patch features
- Compute mean and std → 10D feature vector
- Choose patch size with highest average patch quality

#### 6. Similarity Metrics (`improved_similarity_metrics`)

Three similarity measures combined with weights:

| Metric | Formula | Weight |
|--------|---------|--------|
| Cosine Similarity | (u·v)/(‖u‖‖v‖) | 0.6 |
| Euclidean Similarity | 1/(1 + ‖u-v‖) | 0.25 |
| Manhattan Similarity | 1/(1 + Σ|uᵢ-vᵢ|) | 0.15 |

Final similarity: S = 0.6·S_cosine + 0.25·S_euclidean + 0.15·S_manhattan

#### 7. Evaluation Metrics

- **ROC Curve & AUC** – verification performance
- **Equal Error Rate (EER)** – point where FPR = FNR
- **Decidability Index (d')** – separation between genuine and impostor distributions
- **Identification Accuracy** – 1-to-N matching performance

---

## Fusion Module (In Progress)

The fusion module is currently under development and will combine fingerprint and iris modalities at both feature-level and score-level.

### Proposed Fusion Approaches

#### Case 1: Feature-Level Concatenation
- Concatenate 50D fingerprint vector + 10D iris vector → 60D vector
- Apply fixed orthogonal projection for cancelable templates
- Match using cosine distance

#### Case 2: Score-Level Fusion
- Weighted combination of fingerprint and iris similarity scores
- Dynamic weights based on modality confidence

#### Case 3: Cancelable Template with Reversible Transformation
- Orthogonal random matrix via QR decomposition
- Template = Q × V (where V is concatenated feature vector)
- Distance-preserving transformation (‖T₁ - T₂‖ = ‖V₁ - V₂‖)
- Revocable: generate new matrix if compromised

### Status

| Component | Status |
|-----------|--------|
| Feature extraction (both modalities) | ✅ Complete |
| Individual matching | ✅ Complete |
| Feature-level concatenation | ✅ Complete |
| Score-level fusion | ✅ Complete |
| Cancelable template generation | ✅ Complete |
| Fusion evaluation (identification) | ✅ Complete |
| Fusion evaluation (verification) | ✅ Complete |
| Optimal threshold analysis | ✅ Complete |
| Final tuning and documentation | 🔄 In Progress |

---

## Requirements

### Python Packages

```bash
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
joblib>=1.1.0
Pillow>=8.3.0
```

### Additional for Fingerprint

- `opencv-contrib-python` (for `cv2.ximgproc.thinning`)

### Datasets

- **Fingerprint**: FVC2000 (Db1_a, Db1_b) – requires `.tif` images and `.xyt` minutiae files
- **Iris**: CASIA Iris Image Database (version 1.0) – `.bmp` images organized as `subject/eye/*.bmp`

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi-modal-biometric.git
cd multi-modal-biometric
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For fingerprint thinning support:

```bash
pip install opencv-contrib-python
```

### 3. Dataset Setup

**Fingerprint**:
```
FVC_Databases/
├── FVC2000/
│   ├── Dbs/
│   │   ├── Db1_a/   (image files)
│   │   └── Db1_b/
│   └── xyt/
│       ├── Db1_a/   (minutiae files)
│       └── Db1_b/
```

**Iris**:
```
CASIA Iris Image Database (version 1.0)/
├── 001/
│   ├── 1/   (left eye images)
│   └── 2/   (right eye images)
├── 002/
│   └── ...
```

### 4. Update Paths

In the code, update `DATA_DIR` to your dataset location:

```python
DATA_DIR = r"/path/to/your/Multi-Modal Biometric"
```

---

## Usage

### Running Fingerprint Pipeline

Run the fingerprint cells sequentially (Cells 1-10):

| Cell | Description |
|------|-------------|
| 1 | Imports and configuration |
| 2 | Fingerprint processor class |
| 3 | Data loading and discovery |
| 4 | Feature extraction (triangles, Delaunay) |
| 5 | Matching and score generation |
| 6 | Batch processing pipeline |
| 7 | Score generation with matcher |
| 8 | ROC analysis and optimal threshold |
| 9 | Score distribution visualization |
| 10 | Per-person identification |

### Running Iris Pipeline

Run the iris cells sequentially (Cells 1-6):

| Cell | Description |
|------|-------------|
| 1 | Imports and dataset discovery |
| 2 | Iris localisation and rubber sheet normalisation |
| 3 | GSNE feature extraction |
| 4 | Dataset processing with feature enhancement |
| 5 | Similarity metrics and evaluation |
| 6 | Complete pipeline execution |

### Processing Outputs

**Fingerprint**:
- `enhanced_fingerprint_features.pkl` – extracted triangle features
- `enhanced_fingerprint_matcher.pkl` – trained ML classifier
- `enhanced_identification_threshold_{value}.csv` – per-person identification results

**Iris**:
- `iris_features.npy` – extracted GSNE features (N×10)
- `iris_labels.npy` – corresponding subject labels
- Cached features in `iris_features/` directory

---

## Results

### Fingerprint Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8087 |
| Equal Error Rate (EER) | 28.01% |
| Identification Accuracy | 7.41% |

*Note: Performance is limited by the 50D feature vector requirement for feature-level fusion. The original triangle matcher (variable-length) achieves higher accuracy.*

### Iris Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9999 |
| Equal Error Rate (EER) | 0.14% |
| Identification Accuracy | 100% |

### Fusion Performance (In Progress)

Initial results at threshold 0.85 with 30% fingerprint, 70% iris weights:

| Metric | Value |
|--------|-------|
| ROC-AUC | 1.0000 |
| Equal Error Rate (EER) | 0.10% |
| Identification Accuracy | 99.07% |
| False Acceptance Rate (FAR) | 0.93% |
| False Rejection Rate (FRR) | 0.00% |

---

## Acknowledgments

- **FVC2000** – Fingerprint Verification Competition dataset
- **CASIA Iris Image Database** – Institute of Automation, Chinese Academy of Sciences
- **OpenCV** – Computer vision library for image processing
- **scikit-learn** – Machine learning library for classification and evaluation

---

## License

This project is for academic research purposes.

---

## Contact

For questions or contributions, please contact the project maintainer.

---

## Fusion Module (In Progress)

### Current Status

The fusion module is actively under development and testing. The implementation includes three fusion cases:

#### Case 1: Fingerprint Only
- Baseline performance using 50D fingerprint vectors
- ROC-AUC: 0.8087, EER: 28.01%

#### Case 2: Iris Only
- Baseline performance using 10D GSNE iris features
- ROC-AUC: 0.9999, EER: 0.14%

#### Case 3: Fingerprint + Iris Fusion
- **Feature-Level Concatenation**: 50D fingerprint + 10D iris = 60D vector
- **Cancelable Template**: Orthogonal random matrix (Q) via QR decomposition
  - Template = Q × V (where V is concatenated feature vector)
  - Distance-preserving: ‖T₁ - T₂‖ = ‖V₁ - V₂‖
  - Revocable: generate new matrix if compromised
- **Score-Level Weights**: 30% fingerprint, 70% iris (optimized)
- **Operating Threshold**: 0.85

#### Preliminary Fusion Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 1.0000 |
| EER | 0.10% |
| Identification Accuracy | 99.07% |
| FAR at 0.85 | 0.93% |
| FRR at 0.85 | 0.00% |

#### Cancelable Template Details

The reversible template transformation works as follows:

**Enrollment**:
```
V = [FP_features, Iris_features]  # 60D
T = Q × V  # Cancelable template stored in database
```

**Authentication**:
```
V_live = [FP_live, Iris_live]  # 60D
T_live = Q × V_live
Dist = cosine(T_live, T_db)  # Comparison in transformed domain
```

**Recovery (if needed)**:
```
V_recovered = Q^T × T  # Original features recovered
```

**Properties**:
- Q is orthogonal (Q^T = Q⁻¹)
- Distances preserved in transformed domain
- Without Q, cannot recover original features
- Can be revoked by generating new Q matrix

#### Next Steps

- [ ] Fine-tune fusion weights for optimal performance
- [ ] Implement user-specific tokens (PIN-based matrix generation)
- [ ] Add multi-sample enrollment (averaging multiple templates)
- [ ] Extend evaluation to cross-database testing
- [ ] Complete documentation and final tuning

