import pandas as pd

# 1. Load your original data
df = pd.read_csv('/Users/vipulsharma/Documents/projects/cricket/cricket_features_final3.csv')

def add_follow_on_feature(df):
    follow_on_map = {}
    
    # Group by Match_ID to see the full sequence of innings
    for match_id, match_group in df.groupby('Match_ID'):
        # Get the team batting in each innings (1, 2, 3)
        codes = match_group.groupby('Innings_Num')['Batting_Team_Code'].first()
        
        status = 0
        # A follow-on occurs if the team in Innings 2 also bats in Innings 3
        if 2 in codes.index and 3 in codes.index:
            if codes[2] == codes[3]:
                # If team 2 (Opponent) is batting again, Ref Team enforced it
                if codes[2] == 2:
                    status = 1 
                # If team 1 (Ref Team) is batting again, Opponent enforced it
                else:
                    status = -1
        
        follow_on_map[match_id] = status
        
    # Map the match-level follow-on status to all rows
    df['Follow_On'] = df['Match_ID'].map(follow_on_map)
    
    # PREVENT LEAKAGE: 
    # Set Follow_On to 0 for Innings 1 and 2, because the model 
    # shouldn't "know" it's a follow-on match until it actually happens.
    df.loc[df['Innings_Num'] < 3, 'Follow_On'] = 0
    
    return df

# 2. Process and Save
df_updated = add_follow_on_feature(df)
df_updated.to_csv('cricket_features_with_followon.csv', index=False)

print("Feature added successfully!")
print("Saved to: 'cricket_features_with_followon.csv'")

# Quick verify for a match that had a follow-on
sample_match = 141 
print(df_updated[df_updated['Match_ID'] == sample_match][['Innings_Num', 'Batting_Team_Code', 'Follow_On']].drop_duplicates())