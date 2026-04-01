import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load and Prepare Data
df = pd.read_csv('cricket_features_final3.csv')

# Drop non-statistical features
df_base = df.drop(columns=['Date', 'Venue', 'Reference_Team', 'Opponent'])

le = LabelEncoder()
df_base['Result'] = le.fit_transform(df_base['Result'])

X = df_base.drop(columns=['Result'])
y = df_base['Result']
groups = X['Match_ID']

# 2. Strict Train/Test Split (Match_ID Leakage Protection)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx]

# 3. Load Saved Model and Predict
model = joblib.load('tuned_cricket_model.pkl')
X_test_clean = X_test.drop(columns=['Match_ID'])
y_pred = model.predict(X_test_clean)

# 4. Calculate Overall Metrics
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print("--- Overall Performance Metrics ---")
print(f"{'Metric':<15} | {'Value':<10}")
print("-" * 30)
print(f"{'Accuracy':<15} | {acc:.4f}")
print(f"{'Precision':<15} | {prec:.4f}")
print(f"{'Recall':<15} | {rec:.4f}")
print(f"{'F1-Score':<15} | {f1:.4f}\n")

# 5. Calculate Metrics Per Session (30 overs = 1 session)
results_df = X_test.copy()
results_df['Actual'] = y_test
results_df['Predicted'] = y_pred

# Logic: Session 1 is start of match (450 overs), Session 15 is end (0 overs)
results_df['Session'] = results_df['Overs_Remaining'].apply(lambda x: int(16 - np.ceil(x/30) if x > 0 else 15))

session_data = []
for i in range(1, 16):
    s_subset = results_df[results_df['Session'] == i]
    if len(s_subset) > 0:
        s_acc = accuracy_score(s_subset['Actual'], s_subset['Predicted'])
        s_prec, s_rec, s_f1, _ = precision_recall_fscore_support(
            s_subset['Actual'], s_subset['Predicted'], average='weighted', zero_division=0
        )
        session_data.append([f"Session {i}", s_acc, s_prec, s_rec, s_f1])

# Display Session-wise Table
report_df = pd.DataFrame(session_data, columns=['Timeline', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
print("--- Evaluation Metrics Per Session ---")
print(report_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))