Crop Recommendation System 🌾🚀

📌 Project Overview

This project is a machine learning-based crop recommendation system that helps farmers and agricultural experts determine the best crop to cultivate based on soil and environmental conditions. Using various machine learning models, we identify the most suitable crop based on factors like Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall.

📂 Dataset

The dataset used is from Kaggle: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset?resource=download

Features: N, P, K, Temperature, Humidity, pH, Rainfall
Target: Crop Type (22 different crops including Rice, Maize, Wheat, etc.)

📊 Data Preprocessing

Checked for missing values (None found ✅)

Outlier Removal using Interquartile Range (IQR) method

Label Encoding 

Feature Scaling using StandardScaler

🛠️ Machine Learning Models Used:

The following models were trained and evaluated:
✔ Logistic Regression - 96.13%
✔ K-Nearest Neighbors (KNN) - 97.72%
✔ Decision Tree - 99.54%
✔ Random Forest - 99.77% (Best Model) 🎯
✔ Gradient Boosting - 99.54%
✔ XGBoost - 99.31%
✔ Support Vector Machine (SVM) - 98.18%
✔ Naïve Bayes - 99.54%

📌 Best Model: Random Forest achieved the highest accuracy of 99.77% ✅

🔍 Key Features of the Project
✔ High-accuracy crop prediction based on soil and climate conditions
✔ Uses multiple ML algorithms to determine the best-performing model
✔ Preprocessed & balanced dataset for optimal predictions
✔ Scalable and can be integrated into agriculture-based applications

💻 Tech Stack

🔹 Python 3.11
🔹 Pandas, NumPy, Scikit-learn
🔹 Matplotlib, Seaborn (for visualization)
🔹 Machine Learning (Random Forest, SVM, XGBoost, etc.)
✅ Deploy as a web app using Streamlit
