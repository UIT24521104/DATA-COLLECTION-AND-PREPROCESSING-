from __future__ import annotations
"""
Module: Transformation Visualizations
Chứng minh quá trình biến đổi phân phối và sức mạnh phân tách của Features (Phase 2)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from typing import List
import logging

# Khởi tạo module-level logger
logger = logging.getLogger(__name__)

class TransformationVisualizer:
    """Trực quan hóa quá trình biến đổi dữ liệu thành Phân phối xác suất (KL Divergence Prep)"""
    
    def __init__(self, df_raw: pd.DataFrame, df_transformed: pd.DataFrame, numeric_cols: List[str], custom_logger: logging.Logger = None):
        """
        Khởi tạo với cả dữ liệu gốc (để so sánh) và dữ liệu đã biến đổi.
        """
        self.df_raw = df_raw.copy()
        self.df_transformed = df_transformed.copy()
        self.numeric_cols = numeric_cols
        
        # Ưu tiên dùng custom_logger nếu được truyền vào, nếu không dùng module logger
        self.logger = custom_logger if custom_logger else logger

    def plot_kde_before_after(self, output_folder: Path):
        """1. Biểu đồ KDE: So sánh phân phối Trước (Raw) và Sau (Transformed)"""
        self.logger.info("Generating KDE Before/After plots...")
        
        n_cols = len(self.numeric_cols)
        fig, axes = plt.subplots(nrows=n_cols, ncols=2, figsize=(15, 4 * n_cols))
        
        # Nếu chỉ có 1 cột, axes sẽ là mảng 1D, cần ép về 2D để duyệt chung logic
        if n_cols == 1:
            axes = np.array([axes])
            
        for i, col in enumerate(self.numeric_cols):
            # Cột 1: Dữ liệu gốc (Raw) - Thường bị lệch
            sns.kdeplot(data=self.df_raw[col].dropna(), ax=axes[i, 0], fill=True, color='indianred')
            axes[i, 0].set_title(f'Raw Data: {col}', fontsize=12)
            axes[i, 0].grid(True, alpha=0.3)
            
            # Cột 2: Dữ liệu đã xử lý (Transformed) - Phân phối 0-1
            sns.kdeplot(data=self.df_transformed[col].dropna(), ax=axes[i, 1], fill=True, color='steelblue')
            axes[i, 1].set_title(f'Transformed Data (Prob Dist): {col}', fontsize=12)
            axes[i, 1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(output_folder / 'kde_before_after_transform.png', dpi=120, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: kde_before_after_transform.png")

    def plot_stacked_bar_distributions(self, sample_countries: List[str], output_folder: Path, year: int = 2024):
        """2. Biểu đồ Cột chồng 100%: Chứng minh mỗi quốc gia là một phân phối xác suất"""
        self.logger.info(f"Generating 100% Stacked Bar Chart for year {year}...")
        
        # Kiểm tra xem có cột 'Country Code' và 'Year' không (Thay đổi tên cột cho khớp với Data của bạn)
        country_col = 'Country Code' if 'Country Code' in self.df_transformed.columns else 'Country'
        year_col = 'Year' if 'Year' in self.df_transformed.columns else 'year'
        
        if country_col not in self.df_transformed.columns or year_col not in self.df_transformed.columns:
            self.logger.warning("Missing 'Country Code' or 'Year' columns. Skipping Stacked Bar Chart.")
            return

        df_sample = self.df_transformed[(self.df_transformed[year_col] == year) & 
                                        (self.df_transformed[country_col].isin(sample_countries))]
        
        if df_sample.empty:
            self.logger.warning(f"No data found for year {year} and selected countries. Skipping plot.")
            return

        df_sample = df_sample.set_index(country_col)
        df_plot = df_sample[self.numeric_cols]
        
        # Vẽ biểu đồ cột chồng
        ax = df_plot.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='tab20')
        
        plt.title(f'Macroeconomic Probability Distributions ({year})', fontsize=16, pad=20)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Probability / Weight (Sum = 1.0)', fontsize=12)
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.8) # Đường baseline chuẩn 1.0
        
        # Format legend ra ngoài
        plt.legend(title="Macro Features", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_folder / 'probability_stacked_bars.png', dpi=120, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: probability_stacked_bars.png")

    def plot_tsne_clusters(self, output_folder: Path, perplexity: int = 30):
        self.logger.info("Generating t-SNE Scatter Plot...")
        
        df_clean = self.df_transformed.dropna(subset=self.numeric_cols).copy()
        if len(df_clean) < perplexity:
            self.logger.warning("Not enough data points for t-SNE. Skipping plot.")
            return

        X = df_clean[self.numeric_cols]
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(X)
        
        df_clean['tsne_1'] = tsne_results[:, 0]
        df_clean['tsne_2'] = tsne_results[:, 1]
        
        plt.figure(figsize=(12, 8))
        hue_col = None
        for col in ['Income Group', 'Region', 'Income_Group']:
            if col in df_clean.columns:
                hue_col = col
                break
                
        sns.scatterplot(
            x='tsne_1', y='tsne_2', hue=hue_col, data=df_clean, 
            palette='Set2' if hue_col else None, alpha=0.8, s=60, edgecolor='k'
        )
        
        plt.title('t-SNE Projection: Discriminative Power of Feature Distributions', fontsize=15)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        if hue_col: 
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=hue_col)
            
        plt.tight_layout()
        plt.savefig(output_folder / 'tsne_discriminative_power.png', dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info("Saved: tsne_discriminative_power.png")

    def run_all_visualizations(self, output_folder: str, sample_countries: List[str] = ['VNM', 'THA', 'USA', 'KOR', 'CHN']):
        """Hàm kích hoạt toàn bộ quá trình vẽ biểu đồ Phase 2"""
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("=" * 50)
        self.logger.info("PHASE 2: TRANSFORMATION VISUALIZATIONS")
        self.logger.info("=" * 50)
        
        self.plot_kde_before_after(out_path)
        self.plot_stacked_bar_distributions(sample_countries, out_path)
        self.plot_tsne_clusters(out_path)
        
        self.logger.info("All Phase 2 visualizations completed successfully!")