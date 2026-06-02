from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=8)
def cache_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
