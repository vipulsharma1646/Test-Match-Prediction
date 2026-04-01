import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import joblib # For saving the model

# 1. Load Data
df = pd.read_csv('cricket_features_final3.csv')

# Drop non-statistical features (Original Logic)
df_base = df.drop(columns=['Date', 'Venue', 'Reference_Team', 'Opponent'])

le = LabelEncoder()
df_base['Result'] = le.fit_transform(df_base['Result'])

# Separate features, target, and groups (Match_ID)
X = df_base.drop(columns=['Result'])
y = df_base['Result']
groups = X['Match_ID']

# 2. Train/Test Split (Strictly by Match_ID to prevent leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
groups_train = groups.iloc[train_idx] 

X_test = X.iloc[test_idx].drop(columns=['Match_ID'])
y_test = y.iloc[test_idx]

X_train_clean = X_train.drop(columns=['Match_ID'])

# 3. Hyperparameter Tuning with GridSearchCV
print("Starting Grid Search... Testing multiple parameter combinations.")

base_model = XGBClassifier(eval_metric='mlogloss', random_state=42)

# The combinations of settings to test
param_grid = {
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

# Use GroupKFold to ensure no data leakage during cross-validation
gkf = GroupKFold(n_splits=3)

grid_search = GridSearchCV(
    estimator=base_model, 
    param_grid=param_grid, 
    cv=gkf, 
    scoring='accuracy',
    verbose=1
)

# Fit the grid search 
grid_search.fit(X_train_clean, y_train, groups=groups_train)

# 4. Extract the Best Model
best_model = grid_search.best_estimator_
print(f"\nBest Parameters Found: {grid_search.best_params_}")

# 5. Predict and Evaluate
preds = best_model.predict(X_test)

results_df = X_test.copy()
results_df['Actual'] = y_test
results_df['Predicted'] = preds
results_df['Correct'] = (results_df['Actual'] == results_df['Predicted']).astype(int)

# 6. Plot Accuracy by Overs Remaining
bins = np.arange(0, 451, 45)
results_df['Over_Bin'] = pd.cut(results_df['Overs_Remaining'], bins=bins)
accuracy_by_over = results_df.groupby('Over_Bin')['Correct'].mean() * 100

plt.figure(figsize=(10, 6))
x_labels = [f"{int(b.right)}-{int(b.left)}" for b in accuracy_by_over.index]
plt.plot(x_labels, accuracy_by_over.values, marker='o', linestyle='-', color='g', linewidth=2)

plt.gca().invert_xaxis()
plt.title('Tuned Model Accuracy vs. Match Progression (Original Features)', fontsize=14)
plt.xlabel('Overs Remaining in the Match', fontsize=12)
plt.ylabel('Prediction Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 105)

plt.tight_layout()
plt.savefig('tuned_accuracy_original_features.png')
print("Plot saved as 'tuned_accuracy_original_features.png'.")

# ======================================================
# NEW ADDITIONS: Feature Importance, Confusion Matrix, and Saving
# ======================================================

# 7. Feature Importance
# Identifies which features (Lead, Wickets, etc.) influenced the model most.
importances = best_model.feature_importances_
feature_names = X_train_clean.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)



plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='teal')
plt.title('Feature Importance', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'.")

# 8. Confusion Matrix
# Shows a 3x3 grid comparing actual match results vs. predicted results.
cm = confusion_matrix(y_test, preds)



plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix', fontsize=14)
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'.")

# 9. Save the Model
# "Freezes" the model so you can use it later without retraining.
joblib.dump(best_model, 'tuned_cricket_model.pkl')
print("Model saved as 'tuned_cricket_model.pkl'.")