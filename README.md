Test Match Prediction:--

A Machine Learning Framework for Dynamic Cricket Match Forecasting
This repository contains the codebase for predicting the outcomes (Win, Loss, or Draw) of Test Cricket matches using dynamic, in-play data. Developed as part of a B.S. project at IISER Bhopal, the framework processes ball-by-ball Cricsheet data into 15-over snapshots to capture the complex, evolving nature of the game.


Key Features:--

Multi-Model Architecture: Implements and compares XGBoost, Random Forest, Logistic Regression, and Neural Networks.
Interval Forecasting: Evaluates match dynamics at specific intervals (e.g., every 15 overs) to track model confidence over time.
Advanced Feature Engineering: Incorporates domain-specific metrics, including Relative Required Run Rate and Defensive Stance indicators.


 Repository Structure:--

code/training/: Core machine learning scripts, model training, and evaluation pipelines.

code/preprocessing/: Data cleaning and feature extraction logic.

results/: Performance metrics, confusion matrices, and visualization plots.


 Quick Start:--

1. Clone the repository-->
git clone [https://github.com/vipulsharma1646/Test-Match-Prediction.git](https://github.com/vipulsharma1646/Test-Match-Prediction.git)
cd Test-Match-Prediction

2. Install dependencies-->
pip install -r requirements.txt


3. Run the evaluation script-->
python code/training/comprehensive_5models_evaluation.py


