# Geometric Deep Learning Models: Point Cloud and Graph Classification

This repository contains two machine learning models implemented in Python:
1. **Point Cloud Classification** using a Dynamic Graph Convolutional Neural Network (DGCNN).
2. **Graph Classification** using a GraphSAGE-based model.

## Project Structure
```
.
├── point_cloud_classification.py    # DGCNN for 3D point cloud classification
├── graph_classification.py          # GraphSAGE for graph classification
├── requirements.txt                 # Python dependencies for the project
└── README.md                        # Project documentation
```

---

## 1. Point Cloud Classification (DGCNN)

### Description
This script implements a Dynamic Graph Convolutional Neural Network (DGCNN) to classify 3D point cloud data. It leverages k-Nearest Neighbor (k-NN) graphs dynamically constructed at each layer to learn spatial relationships between points.

### Dataset
- **ModelNet10**: A dataset of 3D object categories represented as point clouds.
- Preprocessing:
  - Samples are downscaled to 1,750 points per object.
  - Point clouds are normalized to fit within a unit sphere.

### Results
- Achieves over 83% test accuracy.
- Visualizes correctly and incorrectly classified point clouds for each class.

### How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python point_cloud_classification.py
   ```

---

## 2. Graph Classification (GraphSAGE)

### Description
This script implements a GraphSAGE model with Graph Convolutional Network (GCNConv) layers for graph classification. The model outputs graph-level predictions with confidence scores.

### Dataset
- Custom graph dataset consisting of:
  - `train.pt`: Training data.
  - `val.pt`: Validation data.
  - `test.pt`: Test data (labels hidden).
  
### Results
- Achieves a validation accuracy of 89.47%.
- Saves predictions with confidence scores to `prediction.csv`.

### How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python graph_classification.py
   ```

---

## Requirements
The dependencies for both tasks are specified in `requirements.txt`.

### Installation
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### Dependencies
- Common:
  - `torch==1.13.0`
  - `torch-geometric==2.1.0`
  - `scikit-learn==1.1.0`
- Point Cloud Classification:
  - `numpy==1.21.6`
  - `scipy==1.7.3`
  - `matplotlib==3.5.1`
- Graph Classification:
  - `pandas==1.3.5`

---

## License
Include your preferred license, e.g., MIT License.
