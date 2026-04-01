import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 1. Load Data
df = pd.read_csv('cricket_features_final3.csv')
df_base = df.drop(columns=['Date', 'Venue', 'Reference_Team', 'Opponent'])

le = LabelEncoder()
df_base['Result'] = le.fit_transform(df_base['Result'])

# Keep Match_ID temporarily for grouping
X = df_base.drop(columns=['Result'])
y = df_base['Result']
groups = X['Match_ID']

# 2. Split by Match_ID (No leakage!)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 3. Drop Match_ID now so the model doesn't use it to train
X_train = X_train.drop(columns=['Match_ID'])
X_test_eval = X_test.drop(columns=['Match_ID'])

# 4. Train the model
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# 5. Predict and Evaluate
preds = model.predict(X_test_eval)

results_df = X_test.copy()
results_df['Actual'] = y_test
results_df['Predicted'] = preds
results_df['Correct'] = (results_df['Actual'] == results_df['Predicted']).astype(int)

# 6. Group by Overs and Plot
bins = np.arange(0, 451, 45)
results_df['Over_Bin'] = pd.cut(results_df['Overs_Remaining'], bins=bins)
accuracy_by_over = results_df.groupby('Over_Bin')['Correct'].mean() * 100

plt.figure(figsize=(10, 6))
x_labels = [f"{int(b.right)}-{int(b.left)}" for b in accuracy_by_over.index]
plt.plot(x_labels, accuracy_by_over.values, marker='o', linestyle='-', color='b', linewidth=2)

plt.gca().invert_xaxis()
plt.title('Corrected Model Accuracy vs. Match Progression', fontsize=14)
plt.xlabel('Overs Remaining in the Match', fontsize=12)
plt.ylabel('Prediction Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 105)

plt.tight_layout()
plt.savefig('corrected_accuracy_by_overs.png')
print("Leakage fixed! Plot saved as 'corrected_accuracy_by_overs.png'.")