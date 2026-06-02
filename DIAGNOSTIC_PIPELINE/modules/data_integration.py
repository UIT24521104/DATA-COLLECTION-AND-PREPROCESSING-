from __future__ import annotations
"""
Module: Data Integration
Tích hợp các file CSV độc lập thành một dataset duy nhất
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import json
# Lưu vết file để kiểm tra sau khi chương trình kết thúc
# Cho phép bật tắt các module khác nhau mà không cần mã nguồn


logger = logging.getLogger(__name__)


class DataIntegration:
    """Tích hợp dữ liệu từ các indicator khác nhau"""
    
    def __init__(self, input_folder: str, logger: logging.Logger = None, feature_mapping_path: str = None):
        self.input_folder = Path(input_folder)
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.merged_data = None
        default_mapping_path = Path(__file__).resolve().parent.parent / 'feature_encoding_rules.json'
        self.feature_mapping_path = Path(feature_mapping_path) if feature_mapping_path else default_mapping_path
        self.feature_name_mapping: Dict[str, str] = {}
        
        # Use provided logger or get module logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def load_feature_mapping(self) -> Dict[str, str]:
        """Load bảng quy tắc mã hóa tên indicator từ file JSON."""
        if not self.feature_mapping_path.exists():
            self.logger.warning(
                f"Feature mapping file not found: {self.feature_mapping_path}. Using original indicator names."
            )
            self.feature_name_mapping = {}
            return self.feature_name_mapping

        try:
            with open(self.feature_mapping_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            mapping = data.get('indicator_mappings', {})
            if not isinstance(mapping, dict):
                raise ValueError("'indicator_mappings' must be a dictionary")

            self.feature_name_mapping = mapping
            self.logger.info(
                f"Loaded {len(self.feature_name_mapping)} feature mappings from {self.feature_mapping_path}"
            )
        except Exception as e:
            self.logger.error(f"Error loading feature mapping file: {e}")
            self.feature_name_mapping = {}

        return self.feature_name_mapping

    def _encode_indicator_name(self, indicator_name: str) -> str:
        """Mã hóa tên indicator theo mapping; nếu không có mapping thì giữ nguyên."""
        encoded_name = self.feature_name_mapping.get(indicator_name, indicator_name)
        if encoded_name == indicator_name and self.feature_name_mapping:
            self.logger.warning(f"No mapping found for indicator: {indicator_name}. Using original name.")
        return encoded_name
    
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load tất cả file CSV từ folder
        - Giả định structure: Quốc gia (rows), năm (columns)
        """
        csv_files = list(self.input_folder.glob('*.csv'))
        # glob là một phương thức tìm kiếm các file dữ liệu theo 1 phương thức nào đó (pattern ví dụ .csv)
        
        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy CSV files ở {self.input_folder}")
            # Lỗi này sẽ được ghi trực tiếp ra màn hình chứ không phải 

        # Load mapping trước để mã hóa tên indicator nhất quán trong toàn bộ pipeline
        self.load_feature_mapping()

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, index_col=0)
                raw_indicator_name = csv_file.stem
                indicator_name = self._encode_indicator_name(raw_indicator_name)

                if indicator_name in self.dataframes:
                    raise ValueError(
                        f"Duplicate encoded indicator name detected: {indicator_name}. "
                        f"Check mapping for collisions."
                    )

                self.dataframes[indicator_name] = df
                self.logger.info(
                    f"Loaded: {raw_indicator_name} -> {indicator_name} - Shape: {df.shape}"
                )
            except Exception as e:
                self.logger.error(f"Error loading {csv_file}: {e}")
        
        return self.dataframes
    
    def standardize_structure(self):
        """
        Chuẩn hóa cấu trúc: quốc gia (rows), years (columns)
        - Sửa kiểu dữ liệu
        - Loại bỏ NaN values rõ ràng
        """
        for key, df in self.dataframes.items():
            # Đảm bảo index là string (quốc gia)
            df.index = df.index.astype(str)
            
            # Đảm bảo columns là numeric (năm)
            # Clean year format: Remove 'YR' prefix before converting to numeric
            df.columns = df.columns.astype(str).str.replace('YR', '').str.strip()
            df.columns = pd.to_numeric(df.columns, errors='coerce')
            
            # Xóa columns không hợp lệ
            df = df.loc[:, df.columns.notna()]
            
            # Convert values thành numeric, NaN cho không phải số
            df = df.apply(pd.to_numeric, errors='coerce')
            
            self.dataframes[key] = df
    
    def pivot_to_long_format(self) -> pd.DataFrame:
        """
        Chuyển từ wide format (Country rows, Years cols) 
        sang long format (Country, Year, Indicator, Value)
        """
        dfs_long = []
        
        for indicator_name, df in self.dataframes.items():
            # Transpose và reset index
            df_long = df.stack().reset_index()
            df_long.columns = ['Country', 'Year', 'Value']
            df_long['Indicator'] = indicator_name
            # Clean year format: "YR2004" -> 2004
            df_long['Year'] = df_long['Year'].astype(str).str.replace('YR', '').str.strip()
            df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
            
            dfs_long.append(df_long)
        
        # Combine tất cả
        combined = pd.concat(dfs_long, ignore_index=True)
        
        return combined
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge các indicators thành một dataset duy nhất
        - Pivot to long format
        - Group by Country & Year, aggregate values
        - Return wide format (Country, Year rows; Indicators columns)
        """
        # Bước 1: Chuyển sang long format
        long_df = self.pivot_to_long_format()
        
        # Bước 2: Aggregate (trong trường hợp có duplicates)
        long_df = long_df.groupby(['Country', 'Year', 'Indicator'])['Value'].mean().reset_index()
        
        # Bước 3: Pivot lại thành wide format (Country+Year rows, Indicators cols)
        # Method: Use pivot instead of pivot_table để tránh aggregation lên lại
        wide_df = long_df.pivot(
            index=['Country', 'Year'],
            columns='Indicator',
            values='Value'
        )
        
        # Bước 4: Reset index để có Country, Year là columns
        wide_df = wide_df.reset_index()
        
        self.merged_data = wide_df
        
        self.logger.info(f"Merged dataset shape: {wide_df.shape}")
        self.logger.info(f"Missing values ratio: {(wide_df.isna().sum().sum() / (wide_df.shape[0] * wide_df.shape[1]) * 100):.2f}%")
        
        return wide_df
    
    def get_data_info(self) -> Dict:
        """Lấy thông tin chung về dữ liệu đã merge"""
        if self.merged_data is None:
            raise ValueError("Chưa merge dữ liệu, gọi merge_datasets() trước")
        
        df = self.merged_data
        
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'years_range': (df['Year'].min(), df['Year'].max()),
            'countries_count': df['Country'].nunique(),
            'records_count': len(df),
            'missing_percentage': (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        }
    
    def run_integration(self) -> pd.DataFrame:
        """Chạy toàn bộ quy trình tích hợp"""
        self.logger.info("=" * 50)
        self.logger.info("PHASE 0: DATA INTEGRATION")
        self.logger.info("=" * 50)
        
        # Step 1: Load
        self.logger.info(f"Step 1: Loading CSV files from {self.input_folder}...")
        self.load_csv_files()
        self.logger.info(f"Loaded {len(self.dataframes)} indicators")
        
        # Step 2: Standardize
        self.logger.info("Step 2: Standardizing structure...")
        self.standardize_structure()
        
        # Step 3: Merge
        self.logger.info("Step 3: Merging datasets...")
        merged = self.merge_datasets()
        
        # Info
        self.logger.info("Integration complete!")
        info = self.get_data_info()
        self.logger.info(f"Data info: {info}")
        
        return merged
