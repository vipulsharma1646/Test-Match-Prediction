import pandas as pd
import numpy as np

df = pd.read_csv('cricket_features_with_followon.csv')

# Calculate Relative RRR (Positive is good for Ref Team, Negative is bad)
df['Relative_RRR'] = np.where(
    df['Innings_Num'] == 4,
    df['Lead'] / df['Overs_Remaining'].replace(0, 1), 
    0  
)

# Drop the old backward-looking Run_Rate
if 'Run_Rate' in df.columns:
    df = df.drop(columns=['Run_Rate'])

df.to_csv('cricket_features_final_rrr.csv', index=False)
print("Added directional 'Relative_RRR'!")