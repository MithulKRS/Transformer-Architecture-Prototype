# 🔥 Fire Guardian Alaska: Spatiotemporal CNN-Transformer Prototype

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-GSoC_2026_Prototype-success?style=for-the-badge)

## 📌 Project Context (GSoC 2026)
This repository contains the architectural prototype for **Fire Guardian Alaska**, a proposed Decision Support System for forecasting wildfire risk using heterogeneous Earth Observation (EO) data. 

While researching the multimodal fusion of high-resolution satellite imagery (Sentinel-1/2) and meteorological data (ERA5), a critical bottleneck was identified: **severe class imbalance**. In Alaska, >95% of the landmass is not burning at any given time. Standard CNN-LSTM models trained with static class weights suffer from "probability squashing," leading to massive over-prediction (False Alarms).

This prototype was engineered prior to the GSoC coding period to validate a **Hybrid CNN-Transformer** architecture and test a custom **Binary Focal Loss** mitigation strategy on simulated $(T, C, H, W)$ spatiotemporal tensors.

---

## 🧠 Architectural Overview

The model bridges localized computer vision with long-range sequence modeling to capture both spatial topologies and temporal drought escalation.

1. **Spatial Encoder (`TimeDistributed` CNN):** Extracts geometric patterns (e.g., contiguous blocks of dry vegetation or topographical slopes) from each time step independently. We utilize Batch Normalization and MaxPooling to stabilize feature extraction.
2. **Temporal Encoder (Transformer Block):** The spatial feature vectors are injected with 1D Positional Embeddings and passed through a Multi-Head Self-Attention mechanism. Unlike LSTMs, this prevents vanishing gradients over long look-back windows (e.g., a 6-month forecast horizon).
3. **Imbalance Mitigation (Focal Loss):** Standard Binary Cross-Entropy is replaced with a custom `BinaryFocalCrossentropy` function to dynamically scale the loss based on prediction confidence, forcing the network to focus on "hard" positives (fires) without artificially inflating the baseline probability of the majority class.

---

## 📊 The Data Pipeline (Synthetic Testbed)

To test the architecture without the heavy overhead of Google Earth Engine (GEE) extractions, this repository includes a deterministic synthetic data generator:
* **Tensor Shape:** `(Batch, Time_Steps, Height, Width, Channels)` -> `(N, 6, 16, 16, 5)`
* **Channels Represented:** `[NDVI, SAR_Moisture, Temperature, Wind_Speed, Humidity]`
* **Hotspot Injection:** The generator injects 4x4 spatial patches of extreme weather conditions (high temp, high wind, low moisture) into the positive labels to give the `Conv2D` layers actual geometric features to learn, mimicking real-world fire clusters.

---

## 🚀 Installation & Execution

### Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install tensorflow numpy scikit-learn
