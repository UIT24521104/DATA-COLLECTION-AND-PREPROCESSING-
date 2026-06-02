import shutil
from pathlib import Path
import pandas as pd

src = Path(r'e:\WorldBank-Data-Analysis-Project\pipeline_execution\outputs\runs\run_20260530_062751_default_phase2\dataset_final.csv')
dst = Path(r'e:\WorldBank-Data-Analysis-Project\pipeline_execution\outputs\latest\dataset_final.csv')

if src.exists():
    shutil.copy2(src, dst)
    print(f'Copied {src.name} to latest/')
    
    # Verify
    df = pd.read_csv(dst)
    print(f'Verified: shape={df.shape}')
    print(f'ABW in dataset: {"ABW" in df["Country"].values}')
    print(f'Unique countries: {df["Country"].nunique()}')
else:
    print('Source file not found')
