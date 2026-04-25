import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import joblib

# 1. Load Data
df = pd.read_csv('/Users/vipulsharma/Documents/projects/cricket/code/csv/cricket_features_with_defense_final.csv')

# Drop metadata
df_base = df.drop(columns=['Date', 'Venue', 'Reference_Team', 'Opponent'])

le = LabelEncoder()
df_base['Result'] = le.fit_transform(df_base['Result'])

X = df_base.drop(columns=['Result'])
y = df_base['Result']
groups = X['Match_ID']

# 2. Train/Test Split (By Match_ID)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train_full = X.iloc[train_idx]
y_train_full = y.iloc[train_idx]
groups_train = groups.iloc[train_idx]

X_test = X.iloc[test_idx].drop(columns=['Match_ID'])
y_test = y.iloc[test_idx]

X_train_clean = X_train_full.drop(columns=['Match_ID'])

# 3. Define Optuna Objective (UPDATED)
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': 1000, # Set artificially high
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'early_stopping_rounds': 50, # Stop if no improvement for 50 trees
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    gkf = GroupKFold(n_splits=3)
    accuracies = []
    
    for t_idx, v_idx in gkf.split(X_train_clean, y_train_full, groups=groups_train):
        X_tr, y_tr = X_train_clean.iloc[t_idx], y_train_full.iloc[t_idx]
        X_va, y_va = X_train_clean.iloc[v_idx], y_train_full.iloc[v_idx]
        
        model = XGBClassifier(**params)
        
        # Pass the validation set so XGBoost can trigger early stopping
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False # Keeps the terminal clean
        )
        
        accuracies.append(accuracy_score(y_va, model.predict(X_va)))
        
    return np.mean(accuracies)

# 4. Run Optimization (UPDATED)
print("Starting Optuna optimization...")
study = optuna.create_study(direction='maximize')

# Increased to 150 to give Optuna room to explore
study.optimize(objective, n_trials=150) 

print(f"\nBest Cross-Validation Accuracy: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")

# 5. Train Final Model (UPDATED)
# Extract the best parameters and ensure early_stopping is set
final_params = study.best_params
final_params['n_estimators'] = 1000
final_params['early_stopping_rounds'] = 50

best_model = XGBClassifier(**final_params, eval_metric='mlogloss', random_state=42)

# For the final model, we can use the test set as the eval_set to trigger early stopping
best_model.fit(
    X_train_clean, y_train_full,
    eval_set=[(X_test, y_test)],
    verbose=False
)
# 6. Final Evaluation
preds = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, preds)

# PRINT FINAL ACCURACY TO TERMINAL
print("-" * 30)
print(f"FINAL TEST ACCURACY: {final_accuracy * 100:.2f}%")
print("-" * 30)

results_df = X_test.copy()
results_df['Actual'], results_df['Predicted'] = y_test, preds
results_df['Correct'] = (results_df['Actual'] == results_df['Predicted']).astype(int)

# 7. Plot: Accuracy vs Overs Remaining
bins = np.arange(0, 451, 15)
results_df['Over_Bin'] = pd.cut(results_df['Overs_Remaining'], bins=bins)
acc_plot = results_df.groupby('Over_Bin', observed=False)['Correct'].mean() * 100

plt.figure(figsize=(14, 7))
x_labels = [f"{int(b.right)}-{int(b.left)}" for b in acc_plot.index]

plt.plot(x_labels, acc_plot.values, marker='o', linestyle='-', color='g', linewidth=2, label='Optuna Tuned Model')

# FIX: Added Baseline
plt.axhline(y=61.11, color='red', linestyle='--', linewidth=2, label='Baseline Accuracy (61.11%)')

# FIX: Set Accuracy Axis (Y-axis) to start from 0
plt.ylim(0, 105)

# FIX: Added Grid Lines
plt.grid(True, linestyle='--', alpha=0.7)

plt.gca().invert_xaxis()
plt.title('Model Accuracy vs. Match Progression (Optuna)', fontsize=14)
plt.xlabel('Overs Remaining in the Match', fontsize=12)
plt.ylabel('Prediction Accuracy (%)', fontsize=12)
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('accuracy_optuna_final4.png')
print("Accuracy plot saved as 'accuracy_optuna_final4.png'.")

# 8. Feature Importance
feat_imp = pd.DataFrame({'Feature': X_train_clean.columns, 'Importance': best_model.feature_importances_}).sort_values('Importance')
plt.figure(figsize=(10, 6))
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='teal')
plt.title('Feature Importance (Optuna)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_optuna4.png')

# 9. Confusion Matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=le.classes_, cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_optuna4.png')

joblib.dump(best_model, 'cricket_model_optuna4.pkl')
print("Model saved as 'cricket_model_optuna4.pkl'.")