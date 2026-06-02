import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data\\terrain_dataset\\labels.csv')

# Stratified split: 70% train, 15% val, 15% test
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df['terrain']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df['terrain']
)

train_df.to_csv(
    'data\\terrain_dataset\\train.csv',
    index=False
)

val_df.to_csv(
    'data\\terrain_dataset\\val.csv',
    index=False
)

test_df.to_csv(
    'data\\terrain_dataset\\test.csv',
    index=False
)

print(
    f'Train: {len(train_df)}, '
    f'Val: {len(val_df)}, '
    f'Test: {len(test_df)}'
)