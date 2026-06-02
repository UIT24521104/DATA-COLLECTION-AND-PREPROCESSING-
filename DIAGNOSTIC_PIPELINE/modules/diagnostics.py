from __future__ import annotations
"""
Module: Diagnostics
Phân tích phân phối, phát hiện outliers, tạo báo cáo chẩn đoán
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
import json

import missingno as msno 

logger = logging.getLogger(__name__)


class DataDiagnostics:
    """Phân tích chẩn đoán dữ liệu"""
    
    def __init__(self, dataframe: pd.DataFrame, logger: logging.Logger = None):
        self.df = dataframe.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.diagnostic_report = {}
        self.outlier_indices = {}
        
        # Use provided logger or get module logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """Phân tích giá trị bị thiếu"""
        missing_report = {
            'total_missing': self.df.isna().sum().sum(),
            'total_records': self.df.shape[0] * self.df.shape[1],
            'missing_percentage': round(
                (self.df.isna().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100, 
                2
            ),
            'columns_missing': {}
        }
        
        for col in self.numeric_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                missing_report['columns_missing'][col] = {
                    'count': int(missing_count),
                    'percentage': round((missing_count / len(self.df)) * 100, 2)
                }
        
        return missing_report
    
    def analyze_distribution(self) -> Dict[str, Any]:
        """Phân tích phân phối các cột numeric"""
        distribution_report = {}
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) == 0:
                continue
            
            distribution_report[col] = {
                'count': len(data),
                'mean': round(float(data.mean()), 4),
                'median': round(float(data.median()), 4),
                'std': round(float(data.std()), 4),
                'min': round(float(data.min()), 4),
                'max': round(float(data.max()), 4),
                'q25': round(float(data.quantile(0.25)), 4),
                'q75': round(float(data.quantile(0.75)), 4),
                'skewness': round(float(data.skew()), 4), # Tính toán hệ số Fisher-Pearson 
                # Skewness = 0 -> Phân phối đối xứng 
                # Skewness > 0 
                # Skewness < 0 
                'kurtosis': round(float(data.kurtosis()), 4), # Tính toán Excess kurtosis
                # k = 0 : Độ nhọn tương đương với phân phối chuẩn 
                # Một đại lượng thống kê đo lường độ nhọn , độ dày của một phân phối xác suất 
                # k > 0 đỉnh rất cao, đuôi nhọn còn, k < 0 thì đỉnh thấp bằng phẳng
            }
        
        return distribution_report
    
    def detect_outliers_iqr(self, multiplier: float = 1.5) -> Dict[str, List[int]]:
        """Phát hiện outliers bằng IQR method"""
        outliers = {}
        
        for col in self.numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            # Tìm outliers
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_indices = self.df[outlier_mask].index.tolist()
            
            if outlier_indices:
                outliers[col] = {
                    'count': len(outlier_indices),
                    'bounds': {
                        'lower': round(float(lower_bound), 4),
                        'upper': round(float(upper_bound), 4),
                    }
                }
                self.outlier_indices[col] = outlier_indices
        
        return outliers
    
    def detect_outliers_zscore(self, threshold: float = 3.0) -> Dict[str, List[int]]:
        """Phát hiện outliers bằng Z-score method"""
        outliers = {}
        
        for col in self.numeric_cols:
            data = self.df[col]
            mean = data.mean()
            std = data.std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((data - mean) / std)
            outlier_mask = z_scores > threshold
            outlier_indices = self.df[outlier_mask].index.tolist()
            
            if outlier_indices:
                outliers[col] = {
                    'count': len(outlier_indices),
                    'threshold': threshold
                }
                self.outlier_indices[f"{col}_zscore"] = outlier_indices
        
        return outliers
    
    def run_diagnostics(self, outlier_method: str = 'iqr', 
                       iqr_multiplier: float = 1.5,
                       zscore_threshold: float = 3.0) -> Dict[str, Any]:
        """Chạy toàn bộ quy trình chẩn đoán"""
        self.logger.info("=" * 50)
        self.logger.info("PHASE 1: AUTO PROFILING & DIAGNOSTICS")
        self.logger.info("=" * 50)
        
        # Step 1: Missing values
        self.logger.info("Step 1: Analyzing missing values...")
        missing_report = self.analyze_missing_values()
        
        # Step 2: Distribution
        self.logger.info("Step 2: Analyzing distributions...")
        distribution_report = self.analyze_distribution()
        
        # Step 3: Outlier detection
        self.logger.info(f"Step 3: Detecting outliers ({outlier_method})...")
        if outlier_method == 'iqr':
            outlier_report = self.detect_outliers_iqr(iqr_multiplier)
        elif outlier_method == 'zscore':
            outlier_report = self.detect_outliers_zscore(zscore_threshold)
        else:
            outlier_report = {}
        
        self.diagnostic_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_shape': self.df.shape,
            'numeric_columns': len(self.numeric_cols),
            'missing_analysis': missing_report,
            'distribution_analysis': distribution_report,
            'outlier_detection': {
                'method': outlier_method,
                'outliers': outlier_report,
                'total_outlier_records': len(set([item for sublist in self.outlier_indices.values() for item in sublist]))
            }
        }
        
        self.logger.info("Diagnostics complete!")
        self.logger.info(f"Found {self.diagnostic_report['outlier_detection']['total_outlier_records']} outlier records")
        
        return self.diagnostic_report
    
    def save_report_csv(self, output_path: str):
        """Lưu báo cáo tổng hợp (Phân phối, Missing, Outliers) dưới dạng CSV"""
        report_data = []
        total_rows = len(self.df)
        
        # Lấy dữ liệu từ các phần của diagnostic_report
        dist_report = self.diagnostic_report.get('distribution_analysis', {})
        missing_report = self.diagnostic_report.get('missing_analysis', {}).get('columns_missing', {})
        outlier_report = self.diagnostic_report.get('outlier_detection', {}).get('outliers', {})

        # Duyệt qua các cột numeric để gộp dữ liệu
        for col in self.numeric_cols:
            # 1. Thông tin phân phối
            stats = dist_report.get(col, {})
            
            row = {
                'Indicator': col,
                **stats,  # mean, median, std, skewness, kurtosis...
            }
            
            # 2. Thông tin Missing Values
            col_missing = missing_report.get(col, {'count': 0, 'percentage': 0.0})
            row['missing_count'] = col_missing['count']
            row['missing_percentage'] = col_missing['percentage']
            
            # 3. Thông tin Outliers
            col_outlier = outlier_report.get(col, {'count': 0, 'bounds': {'lower': None, 'upper': None}})
            row['outlier_count'] = col_outlier['count']
            row['outlier_percentage'] = round((col_outlier['count'] / total_rows) * 100, 2) if total_rows > 0 else 0.0
            row['outlier_lower_bound'] = col_outlier['bounds']['lower']
            row['outlier_upper_bound'] = col_outlier['bounds']['upper']
            
            report_data.append(row)
        
        if report_data:
            df_report = pd.DataFrame(report_data)
            # Sắp xếp lại thứ tự cột cho dễ nhìn
            cols_order = ['Indicator', 'count', 'missing_count', 'missing_percentage', 
                          'outlier_count', 'outlier_percentage', 'mean', 'median', 'std', 'skewness', 'kurtosis']
            # Giữ lại các cột khác (min, max, q25, q75...) ở phía sau
            remaining_cols = [c for c in df_report.columns if c not in cols_order]
            df_report = df_report[cols_order + remaining_cols]
            
            df_report.to_csv(output_path, index=False, encoding='utf-8-sig') # Thêm utf-8-sig để mở Excel không lỗi font
            self.logger.info(f"Full diagnostic report saved to {output_path}")

    def plot_correlation_heatmap(self, output_folder: Path, figsize: Tuple[int, int] = (12, 10)):
        """Vẽ Heatmap tương quan giữa các biến"""
        self.logger.info("Generating Heatmap...")
        plt.figure(figsize=figsize)
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Che nửa trên ma trận cho đỡ rối mắt
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            vmin=-1, vmax=1, 
            square=True, 
            linewidths=.5,
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Correlation Heatmap', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(Path(output_folder) / 'correlation_heatmap.png', dpi=120, bbox_inches='tight')
        self.logger.info("Saved: correlation_heatmap.png")
        plt.close()

    def plot_pairwise_scatter(self, output_folder: Path):
        """Vẽ Scatter Plot ma trận (Chọn top 5 biến để tránh quá tải RAM)"""
        self.logger.info("Generating Pairwise Scatter Plot...")
        
        # Chọn tối đa 5 biến để vẽ cho nhanh và dễ nhìn
        sample_cols = self.numeric_cols[:5] if len(self.numeric_cols) > 5 else self.numeric_cols
        df_plot = self.df[sample_cols].dropna()
        
        pair_plot = sns.pairplot(
            df_plot, 
            kind='scatter', 
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'none'}
        )
        
        pair_plot.fig.suptitle('Pairwise Scatter Plot (Top Features)', y=1.02, fontsize=16)
        pair_plot.savefig(Path(output_folder) / 'pairwise_scatter.png', dpi=150, bbox_inches='tight')
        self.logger.info("Saved: pairwise_scatter.png")
        plt.close()

# API của missingno không hỗ trợ hàm savefig() để lưu lại ảnh 
    def plot_matrix_plot(self, output_folder: Path):
        """Vẽ matrix Plot để phát hiện ra những quy luật ẩn trong việc phân tích dữ liệu"""
        self.logger.info("Generating Matrix Plot of missingno...")
        data = self.df

        matrix_plot = msno.matrix(data)
        plt.savefig(Path(output_folder)/ 'matrix_plot.png', dpi = 150, bbox_inches = 'tight')
        plt.close()
        self.logger.info("Saved: Matrix Plot")
    
    def plot_heatmap_msn(self, output_folder: Path):
        """Vẽ biểu đồ nhiệt đối miêu tả tương quan dữ liệu thiếu giữa các features"""
        self.logger.info("Generating Heatmap Plot of missingno...")
        data = self.df
        heatmap_plot = msno.heatmap(data)

        plt.savefig(Path(output_folder)/'heatmap_plot.png', dpi = 150, bbox_inches = 'tight')
        plt.close()
        self.logger.info("Saved: Heatmap Plot")

    def create_visualizations(self, output_folder: str, figsize: Tuple[int, int] = (15, 10)):
        """Tạo visualizations: boxplots, histograms, heatmap, scatter plots"""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Missing values theo thời gian (nếu có cột Year)
        if 'Year' in self.df.columns:
            indicator_cols = [col for col in self.df.columns if col not in ['Country', 'Year']]
            if indicator_cols:
                missing_by_year = (
                    self.df.groupby('Year')[indicator_cols]
                    .apply(lambda group: group.isna().mean().mean() * 100)
                    .sort_index()
                )

                plt.figure(figsize=(12, 5))
                plt.plot(missing_by_year.index, missing_by_year.values, marker='o', linewidth=2)
                plt.title('Missing Values by Year (%)', fontsize=12)
                plt.xlabel('Year')
                plt.ylabel('Missing Percentage (%)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_folder / 'missing_values_by_year.png', dpi=120, bbox_inches='tight')
                self.logger.info("Saved: missing_values_by_year.png")
                plt.close()

                # Missing values theo từng feature qua các năm (nhiều subplot)
                missing_by_year_feature = (
                    self.df.groupby('Year')[indicator_cols]
                    .apply(lambda group: group.isna().mean() * 100)
                    .sort_index()
                )

                n_features = len(indicator_cols)
                ncols = 2
                nrows = (n_features + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows), squeeze=False)
                axes = axes.flatten()

                for idx, feature in enumerate(indicator_cols):
                    ax = axes[idx]
                    ax.plot(
                        missing_by_year_feature.index,
                        missing_by_year_feature[feature].values,
                        marker='o',
                        linewidth=1.8,
                    )
                    ax.set_title(feature, fontsize=10)
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Missing %')
                    ax.grid(True, alpha=0.3)

                # Xóa subplot trống nếu số feature lẻ
                for idx in range(n_features, len(axes)):
                    fig.delaxes(axes[idx])

                plt.tight_layout()
                plt.savefig(output_folder / 'missing_values_by_feature_by_year.png', dpi=130, bbox_inches='tight')
                self.logger.info("Saved: missing_values_by_feature_by_year.png")
                plt.close()
        
        # Boxplots
        fig, axes = plt.subplots(nrows=(len(self.numeric_cols) + 1) // 2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(self.numeric_cols):
            axes[idx].boxplot(self.df[col].dropna())
            axes[idx].set_title(f'Boxplot: {col}', fontsize=10)
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3)
        
        # Xóa subplot trống
        for idx in range(len(self.numeric_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(output_folder / 'boxplots.png', dpi=100, bbox_inches='tight')
        self.logger.info("Saved: boxplots.png")
        plt.close()
        
        # Histograms
        fig, axes = plt.subplots(nrows=(len(self.numeric_cols) + 1) // 2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(self.numeric_cols):
            axes[idx].hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Histogram: {col}', fontsize=10)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Xóa subplot trống
        for idx in range(len(self.numeric_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(output_folder / 'histograms.png', dpi=100, bbox_inches='tight')
        self.logger.info("Saved: histograms.png")
        plt.close()



        #Correlation heatmap
        self.plot_correlation_heatmap(output_folder)

        #Pairwise scatter plots
        self.plot_pairwise_scatter(output_folder) 

        self.plot_matrix_plot(output_folder)

        self.plot_heatmap_msn(output_folder)


