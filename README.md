
🚀 Financial Budgeting & Forecasting with LSTM
A Machine Learning Proof of Concept for Predicting Monthly Net Cash Flow

📌 Overview
This repository contains the full implementation, datasets, models, notebooks, and results for a financial forecasting proof of concept. The goal of this project was to determine whether machine learning—specifically Long Short-Term Memory (LSTM) networks—can accurately predict a household’s next-month net cash flow using historical expenditure and income data.
This work was completed as part of the NYU Applied Project Capstone, demonstrating how advanced neural networks can support adaptive budgeting tools and AI-driven financial planning applications.

🎯 Key Features
Complete data engineering pipeline using U.S. Consumer Expenditure Survey (CES) microdata
Preprocessing, transformation, and normalization of raw financial datasets
Implementation of Linear Regression, Feedforward Neural Network (FFNN), and LSTM models
Full model comparison using MAE, RMSE, and R² metrics
Visualization of training curves, residuals, and actual vs. predicted forecasts
Saved models and scalers for reproducibility
Clean, professional repository structure
Git LFS support for large datasets and model files

🧠 Problem Statement
Traditional budgeting tools summarize past spending but cannot predict future cash flow.
This project asks:
Can machine learning models accurately forecast monthly net cash flow using historical income and expenditure patterns?

📘 Research Approach
Data Engineering
Transform CES microdata from quarterly → monthly
Aggregate expenditure categories
Normalize and scale numeric fields
Create supervised learning sequences (time windows)
Model Development
Build Linear Regression model (baseline)
Build FFNN for nonlinear modeling
Build LSTM to capture temporal dependencies
Model Evaluation
Compare MAE, RMSE, R²
Generate forecast visualizations
Analyze error patterns
Documentation & Reproducibility
Organized datasets and outputs
Published models + code
Fully documented methodology

📊 Results Summary
Model	MAE ($)	RMSE ($)	R²
Linear Regression	123.87	198.44	0.99994
FFNN	107.32	174.91	0.99996
LSTM	68.42	109.77	0.99998
✔ LSTM outperformed all other models
✔ Forecasts followed real data closely
✔ Residuals were stable and unbiased
✔ Strong evidence that machine learning can support financial forecasting tools

📂 Repository Structure
financial-budgeting-forecasting/
│
├── data_raw/                 # Raw CES microdata
├── data_processed/           # Cleaned and transformed datasets
├── models/                   # Trained model files (.keras)
├── results/                  # PNG charts and graphs
├── notebooks/                # Jupyter notebooks (EDA, modeling)
└── README.md

🔍 Notebooks Included
EDA.ipynb
Exploratory data analysis, summary statistics, visualization.
Model_Training.ipynb
Full implementation of Linear Regression, FFNN, and LSTM models.
Evaluation.ipynb
Generates charts, metrics, comparison tables, and predictions.

📈 Visual Outputs
(Have these ready in /results; you can embed them later if you want)
LSTM prediction vs actual
LSTM loss curve
FFNN loss curve
Model comparison MAE
Residual analysis
Training curve visualizations

🛠️ Technologies Used
Python
NumPy
Pandas
TensorFlow / Keras
Scikit-Learn
Matplotlib / Seaborn
Git & Git LFS

🧪 How to Run This Project
1. Clone the repo
git clone https://github.com/YOUR_USERNAME/financial-budgeting-forecasting.git
2. Install dependencies
pip install -r requirements.txt
(If you want, I can generate this file for you)
3. Run the notebooks
Open Jupyter or Colab and run the files inside /notebooks.

📁 Datasets and Models
All datasets and model files are stored in:
/data_raw
/data_processed
/models
Git LFS is enabled to support large files.

👤 Author
Vishwa Shah
Master’s Student, NYU SPS
Financial Modeling | Machine Learning | Data Engineering
LinkedIn: https://www.linkedin.com/in/vishwadipeshshah
Email: vishwashah.nyu@gmail.com

⭐ Future Work
Integrating a financial assistant chatbot using LLMs
Testing Liquid Neural Networks (LNNs) for adaptive forecasting
Deploying the model in a budgeting dashboard
Adding multi-step forecasting (predicting several months ahead)
