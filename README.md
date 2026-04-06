# Multimodal COVID-19 Detection via Intermediate Fusion 🚀

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

## 📌 Project Overview
Early and accurate detection of COVID-19 is critical for effective treatment. This project implements a **Multimodal Deep Learning** approach that integrates three distinct data sources: **Computed Tomography (CT)**, **Chest X-Ray (CXR)**, and **Cough Audio** signals to enhance diagnostic precision.

---

## 📂 Data Pipeline & Preprocessing
We handled three heterogeneous data modalities with specific preprocessing pipelines:

* **CT & CXR Images:** Resized to **224x224x3** and normalized to $[0, 1]$.
* **Cough Audio:** Raw signals were converted into **Mel-Spectrograms** (2D frequency-pattern representations) to be processed by CNN layers.

---

## 🧠 Model Architecture: The 3-Stream Approach
The **Intermediate Fusion** model (Our Best Model) uses three independent feature extraction branches:

1.  **CT Branch:** ResNet50V2 backbone for spatial features.
2.  **CXR Branch:** Parallel ResNet50V2 for lung opacity detection.
3.  **Audio Branch:** Custom CNN for spectral pattern recognition in coughs.

**Fusion Mechanism:** Branch outputs are concatenated and passed through a series of **Dense Layers** ($512 \rightarrow 256 \rightarrow 128$) with **Batch Normalization** and **Dropout (0.5)** to ensure robust classification.



---

## 📊 Experimental Results
We conducted a rigorous comparison to justify the choice of architecture:

| Fusion Strategy | Accuracy | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| Early Fusion (9-Channel) | 87.5% | 0.86 | 0.87 | 0.87 |
| **Intermediate Fusion (Proposed)** | **94.2%** | **0.94** | **0.94** | **0.94** |

### 📈 Training Optimization
* **Optimizer:** Adam ($1 \times 10^{-4}$).
* **Callbacks:** `ReduceLROnPlateau` and `EarlyStopping` (Patience: 5) to ensure optimal weight restoration.

---

## 🛠️ Tech Stack
- **Frameworks:** TensorFlow, Keras (Functional API).
- **Backbones:** ResNet50V2.
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn.

---
