
📌 Project Overview

This repository contains a complete end-to-end machine learning pipeline for predicting weather conditions using contextual metadata only (no image pixels).

The task is formulated as a supervised multi-class classification problem, where each sample is classified into one of four weather categories:

Sunny · Cloudy · Rainy · Snowy

based on the following contextual attributes:

Country

Season

Time of Day

The project emphasizes data quality, class imbalance handling, and error analysis, and demonstrates both the strengths and limitations of metadata-based prediction.

⭐ Key Highlights

✔ Full data pipeline from raw files to trained models
✔ Robust data cleaning and normalization
✔ Real-world image URL validation and downloading
✔ Exploratory Data Analysis (EDA) with visualizations
✔ Baseline vs advanced ML model comparison
✔ Class imbalance-aware evaluation (weighted F1-score)
✔ Detailed error analysis with saved misclassifications

🗂️ Repository Data Files (Important)
📄 combined_dataset.csv

Purpose:
This file contains all raw data merged into a single dataset after reading multiple CSV and Excel files with inconsistent formats.

Key characteristics:

Unified schema across all sources

Contains noisy, missing, and inconsistent values

Includes all original rows before strict filtering

Used as the starting point for EDA and preprocessing

📌 This file allows transparency and reproducibility of the full data-cleaning process.

📄 successful_rows.csv

Purpose:
This file represents the final cleaned dataset with successfully downloaded images, and is the main dataset used for modeling.

How it was created:

Rows with invalid URLs, missing required fields, or ambiguous labels were removed

Each remaining sample was assigned a stable ID

Images were downloaded using the provided URLs

Only rows with successfully downloaded images were kept

Key characteristics:

Clean and consistent labels

Strongly reduced noise

Direct one-to-one mapping between dataset rows and image files

Used for:

Exploratory Data Analysis

Model training and evaluation

Error analysis

📊 Final dataset size: ~676 samples

🔍 Exploratory Data Analysis (EDA)

EDA was performed mainly on the cleaned datasets to understand:

Missing value percentages

URL validity

Category distributions

Feature relationships

Visualizations include:

Bar charts of feature frequencies

Heatmaps:

Weather vs Season

Weather vs Time of Day

📉 Findings:

Severe class imbalance (Sunny dominates, Rainy is rare)

Snowy shows strong seasonal patterns

High overlap between Sunny and Cloudy

🧠 Machine Learning Pipeline
🔹 Baseline Model

K-Nearest Neighbours (KNN)

Tested with k = 1 and k = 3

Best baseline selected using weighted F1-score

🔹 Advanced Models

Support Vector Machine (RBF Kernel)

Class-balanced

Hyperparameter tuning with Grid Search

Random Forest

Ensemble-based

Handles feature interactions well

Achieved the highest weighted F1-score on the test set

📏 Evaluation Metrics

Because the dataset is imbalanced, the following metrics were used:

Weighted F1-score (primary metric)

Accuracy

Confusion matrices

Error rate per true class

🧪 Error Analysis

A detailed error analysis was performed on the best-performing model:

Misclassified samples were saved to a separate CSV file

Error rates were analyzed per:

Weather class

Country

Season

Time of Day

🔎 Key Insights

Rainy has the highest error rate due to extreme underrepresentation

Snowy is classified perfectly due to strong contextual signals

Most confusion occurs between Sunny ↔ Cloudy

Metadata alone is insufficient to fully separate visually similar weather conditions

🚀 Conclusions

Metadata-based weather classification is feasible but limited

Class imbalance significantly affects minority classes

Advanced models outperform simple baselines in balanced metrics

Future improvements should include:

More balanced datasets

Integration of image-based (computer vision) features
