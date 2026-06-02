import pandas as pd
import sys

# Force fresh import
file_path = r'e:\WorldBank-Data-Analysis-Project\pipeline_execution\outputs\latest\dataset_final.csv'
print(f"Reading from: {file_path}")

df = pd.read_csv(file_path)
print(f"\nFinal dataset shape: {df.shape}")
print(f"Countries in final: {df['Country'].nunique()}")
print(f"Has ABW: {'ABW' in df['Country'].values}")
print(f"\nFirst country: {df['Country'].iloc[0]}")
print(f"Unique countries count: {len(df['Country'].unique())}")
