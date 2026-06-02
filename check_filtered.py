import pandas as pd

# Check diagnostic report
diag = pd.read_csv(r'e:\WorldBank-Data-Analysis-Project\pipeline_execution\outputs\latest\diagnostic_report.csv')
print("Diagnostic report shape:", diag.shape)

# Check merged dataset
merged = pd.read_csv(r'e:\WorldBank-Data-Analysis-Project\pipeline_execution\outputs\latest\dataset_merged.csv')
print(f"Merged dataset shape: {merged.shape}")
print(f"Countries in merged: {merged['Country'].nunique()}")
print(f"Has ABW in merged: {'ABW' in merged['Country'].values}")

# Check final dataset
final = pd.read_csv(r'e:\WorldBank-Data-Analysis-Project\pipeline_execution\outputs\latest\dataset_final.csv')
print(f"\nFinal dataset shape: {final.shape}")
print(f"Countries in final: {final['Country'].nunique()}")
print(f"Has ABW in final: {'ABW' in final['Country'].values}")

print(f"\nNon-countries that should be removed:")
non_countries = ['HKG', 'MAC', 'PRI', 'GRL', 'BMU', 'VIR', 'VGB', 'CYM', 'TCA', 'ABW', 'CUW', 'SXM', 'MAF',
    'FRO', 'GIB', 'IMN', 'CHI', 'GUM', 'NCL', 'PYF', 'MNP', 'ASM', 'PSE', 'XKX',
    'WLD', 'HIC', 'UMC', 'MIC', 'LMC', 'LIC', 'LDC', 'HPC', 'FCS', 'IDA', 'IBD', 'IBT', 'IDX', 'IDB',
    'EAR', 'PRE', 'PST', 'SST', 'OSS', 'EUU', 'EMU', 'OED', 'ARB',
    'AFE', 'AFW', 'SSF', 'SSA', 'LCN', 'LAC', 'TLA', 'NAC', 'CSS',
    'ECS', 'ECA', 'TEC', 'CEB', 'EAS', 'EAP', 'TEA', 'SAS', 'TSA', 'PSS', 'MEA', 'MNA', 'TMN']

found_in_final = [c for c in non_countries if c in final['Country'].values]
print(f"  Still in final: {found_in_final}")

