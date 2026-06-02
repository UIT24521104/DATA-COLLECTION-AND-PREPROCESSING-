"""
Thay vì chạy cả project lớn thì chỉ chạy đúng 1 file này và kiểm tra có lỗi hay không
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'modules'))

from data_integration import DataIntegration

# Use absolute path from project root
input_folder = str(Path(__file__).parent.parent / 'nam')

print("Starting data integration...")
integration = DataIntegration(input_folder=input_folder)

print("\n1. Loading CSVs...")
dataframes = integration.load_csv_files()
print(f"   Loaded {len(dataframes)} indicators")

print("\n2. Standardizing structure...")
integration.standardize_structure()
print(f"   First indicator shape: {list(integration.dataframes.values())[0].shape}")
print(f"   First indicator sample:")
df_sample = list(integration.dataframes.values())[0]
print(df_sample.iloc[:2, :3])

print("\n3. Converting to long format...")
long_df = integration.pivot_to_long_format()
print(f"   Long format shape: {long_df.shape}")
print(f"   Columns: {long_df.columns.tolist()}")
print(f"   Year range: {long_df['Year'].min()} - {long_df['Year'].max()}")
print(f"   Sample (first 5 rows):")
print(long_df.head())

print("\n4. Merging datasets...")
merged_df = integration.merge_datasets()
print(f"   Merged shape: {merged_df.shape}")
print(f"   Columns: {merged_df.columns.tolist()}")
if len(merged_df) > 0:
    print(f"   Sample (first 5 rows):")
    print(merged_df.head())
else:
    print("   ⚠️  Empty dataframe!")

print("\nDone!")
