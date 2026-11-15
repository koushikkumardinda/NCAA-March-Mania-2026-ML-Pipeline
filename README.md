# 🏀 March Machine Learning Mania 2026: Advanced Elo & Meta-Ensemble Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20beff.svg)](https://www.kaggle.com/)

## 📖 Overview
This repository contains a complete predictive modeling pipeline developed for the Kaggle **March Machine Learning Mania 2026** competition. Moving beyond simple win rates, this project implements a professional sports analytics workflow designed to capture strength of schedule, late-season momentum, and core basketball fundamentals to accurately predict tournament upsets.

## 🧠 Methodology & Approach
The predictive engine relies on three core pillars:

1. **Dynamic Elo Ratings:** Calculates season-long running Elo ratings for every Division I team. This naturally captures strength of schedule and late-season momentum far better than static win percentages.
2. **Advanced Metrics (Four Factors):** Leverages play-by-play Detailed Results to calculate True Shooting Percentage (TS%) and Rebound Margins, simulating Dean Oliver's "Four Factors" of basketball success.
3. **Calibrated Meta-Ensemble:** Trains both **LightGBM** and **XGBoost** models on a symmetric target dataset. The models' predicted probabilities are blended to minimize the Brier Score and Log Loss (Kaggle's primary evaluation metrics).

## 🛠️ Tech Stack & Requirements
* **Language:** Python 3.12
* **Data Manipulation:** `numpy`, `pandas`
* **Machine Learning:** `scikit-learn`, `lightgbm`, `xgboost`
* **Visualization:** `matplotlib`, `seaborn`

## 🔀 Feature Engineering Highlights
To ensure the model learns absolute team strength rather than target-leakage bias, the training set is built as a **symmetric matrix**. 
Predictive features include:
* Seed Differences
* Elo Rating Differences
* True Shooting Percentage (TS%) Deltas
* Rebound Margin Deltas
* Massey Ordinal Rank Differences (Men's Tournament)

## 🚀 Usage
1. Ensure the Kaggle dataset is downloaded and extracted to an `input/` directory.
2. Run the Jupyter Notebook environment to generate the advanced features, train the LightGBM/XGBoost folds, and visualize feature importance.
3. The pipeline outputs a final `submission.csv` with predictions clipped between `0.01` and `0.99` for Log Loss safety.
