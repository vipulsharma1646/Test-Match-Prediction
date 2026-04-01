import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# 1. Load the data
# Make sure the CSV file is in the same directory as this script
df = pd.read_csv('cricket_features_final3.csv')

# 2. Initial Cleanup & Target Encoding
# Drop columns that are pure identifiers and have no predictive value
df_base = df.drop(columns=['Match_ID', 'Date'])

# XGBoost requires target classes to start from 0 (e.g., 0, 1, 2)
# LabelEncoder handles this automatically (e.g., mapping -1, 0, 1 to 0, 1, 2)
le = LabelEncoder()
df_base['Result'] = le.fit_transform(df_base['Result'])

print(f"Classes mapped as: {dict(zip(le.classes_, range(len(le.classes_))))}\n")

# ==========================================
# VERSION 1: Name-Biased Model (One-Hot Encoded)
# ==========================================
print("--- Training Version 1: With Team Names & Venues ---")

# One-Hot Encode the categorical text columns
df_v1 = pd.get_dummies(df_base, columns=['Venue', 'Reference_Team', 'Opponent'])

# Separate features (X) and target (y)
X1 = df_v1.drop(columns=['Result'])
y1 = df_v1['Result']

# Split into 80% training and 20% testing data
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model_v1 = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_v1.fit(X1_train, y1_train)

# Predict and evaluate
preds_v1 = model_v1.predict(X1_test)
acc_v1 = accuracy_score(y1_test, preds_v1)
print(f"Version 1 Accuracy: {acc_v1 * 100:.2f}%\n")


# ==========================================
# VERSION 2: Pure Statistical Model (Dropped Names)
# ==========================================
print("--- Training Version 2: Pure Statistical Features Only ---")

# Drop the categorical text columns entirely
df_v2 = df_base.drop(columns=['Venue', 'Reference_Team', 'Opponent'])

# Separate features (X) and target (y)
X2 = df_v2.drop(columns=['Result'])
y2 = df_v2['Result']

# Split using the exact same random state to ensure an apples-to-apples comparison
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model_v2 = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_v2.fit(X2_train, y2_train)

# Predict and evaluate
preds_v2 = model_v2.predict(X2_test)
acc_v2 = accuracy_score(y2_test, preds_v2)
print(f"Version 2 Accuracy: {acc_v2 * 100:.2f}%\n")


# ==========================================
# RESULTS COMPARISON
# ==========================================
print("--- Final Comparison ---")
print(f"Model 1 (With Names): {acc_v1 * 100:.2f}%")
print(f"Model 2 (No Names):   {acc_v2 * 100:.2f}%")

if acc_v1 > acc_v2:
    print("\nConclusion: The model relies slightly on specific team/venue biases to get a higher accuracy.")
else:
    print("\nConclusion: The pure statistical features are strong enough on their own. The model generalizes well without needing to know exactly who is playing.")