import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

# 1. Load Data
df = pd.read_csv('/Users/vipulsharma/Documents/projects/cricket/cricket_features_final3.csv')

# 2. Encode the Target Variable (so Win/Draw/Loss become numbers)
le = LabelEncoder()
df['Encoded_Result'] = le.fit_transform(df['Result'])

# 3. Get only ONE row per match (so we don't overcount overs)
matches = df.drop_duplicates(subset=['Match_ID']).copy()

# 4. Calculate the Baseline (Most frequent outcome per matchup)
baseline_logic = matches.groupby(['Reference_Team', 'Opponent'])['Encoded_Result'].apply(lambda x: x.mode()[0]).reset_index()
baseline_logic.rename(columns={'Encoded_Result': 'Baseline_Prediction'}, inplace=True)

# 5. Merge the predictions back into the matches dataframe
matches = matches.merge(baseline_logic, on=['Reference_Team', 'Opponent'], how='left')

# 6. Recreate your EXACT Test Set using the same random_state=42
groups = df['Match_ID']
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, df['Encoded_Result'], groups))

# Isolate the matches that belong to the Test Set
test_match_ids = groups.iloc[test_idx].unique()
test_matches = matches[matches['Match_ID'].isin(test_match_ids)].copy()

# 7. Calculate Final Test Accuracy
baseline_accuracy = (test_matches['Encoded_Result'] == test_matches['Baseline_Prediction']).mean() * 100
print(f"Test Set Baseline Accuracy: {baseline_accuracy:.2f}%")