import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
df = pd.read_csv('cricket_features_final3.csv')

# Drop non-statistical features
df_base = df.drop(columns=['Date', 'Venue', 'Reference_Team', 'Opponent'])
le = LabelEncoder()
df_base['Result'] = le.fit_transform(df_base['Result'])

X = df_base.drop(columns=['Result'])
y = df_base['Result']
groups = X['Match_ID']

# 2. Strict Split (Train vs Test)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# 3. Train Model ONLY on Training Data
X_train_clean = X_train.drop(columns=['Match_ID'])
model = RandomForestClassifier(random_state=42)
model.fit(X_train_clean, y_train)

# 4. Pick a match strictly from the UNSEEN Test Data
# Grabbing the first unique match ID in the test set
target_match_id = X_test['Match_ID'].unique()[0]

# Extract info for the title
ref_team = df[df['Match_ID'] == target_match_id]['Reference_Team'].iloc[0]
opp_team = df[df['Match_ID'] == target_match_id]['Opponent'].iloc[0]

match_data = X_test[X_test['Match_ID'] == target_match_id].copy()
match_data = match_data.sort_values('Overs_Remaining', ascending=False)

X_match = match_data.drop(columns=['Match_ID'])
overs_timeline = match_data['Overs_Remaining']

# 5. Generate Probabilities
probabilities = model.predict_proba(X_match)
prob_loss = probabilities[:, 0] * 100
prob_draw = probabilities[:, 1] * 100
prob_win  = probabilities[:, 2] * 100

# 6. Plot the Timeline
plt.figure(figsize=(12, 6))

plt.plot(overs_timeline, prob_win,  label='Win Probability',  color='green', linewidth=2)
plt.plot(overs_timeline, prob_loss, label='Loss Probability', color='red',   linewidth=2)
plt.plot(overs_timeline, prob_draw, label='Draw Probability', color='blue',  linewidth=2, linestyle='--')

plt.gca().invert_xaxis()
plt.title(f' Match Timeline: {ref_team} vs {opp_team} (Match ID: {target_match_id})', fontsize=16, fontweight='bold')
plt.xlabel('Overs Remaining in the Match', fontsize=12)
plt.ylabel('Probability (%)', fontsize=12)
plt.ylim(0, 105)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best', fontsize=11)

# Adding the innings change markers
innings_changes = match_data['Innings_Num'].diff().fillna(0)
innings_change_overs = match_data[innings_changes != 0]['Overs_Remaining'].tolist()

for change_over in innings_change_overs:
    plt.axvline(x=change_over, color='gray', linestyle=':', alpha=0.8)
    plt.text(change_over, 102, 'Innings Change', rotation=90, verticalalignment='top', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(f'unseen_timeline_match_{target_match_id}.png')
print(f"Success! Unseen Match Timeline saved as 'unseen_timeline_match_{target_match_id}.png'")