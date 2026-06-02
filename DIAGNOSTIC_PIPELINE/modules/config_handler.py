"""
Module: Config Handler
Quản lý cấu hình YAML cho pipeline
"""
# Handler : một hàm hoặc phương thức được thiết kế để đáp trả một sự kiện hoặc điều kiện cụ thể xảy ra trong chương trình
# Handler nhận đầu vào là file log đầu ra là action


import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass # decorator tạo một lớp dữ liệu chuyên biệt 
# Giúp xem class  def được sử dụng lại nhiều lần ma không cần code  lại
# TH1: Truyền vào function 1 function khác 
# TH2: Biến đổi 1 class bình thường thành 1 DataClass để chứa dữ liệu mà khôgn cần thiết để viết các phương thức bổ sung 
class OutlierConfig:
    """Cấu hình xử lý Outlier"""
    method: str = "iqr"  # iqr, zscore, isolation_forest
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    isolation_forest_contamination: float = 0.05
    action: str = "clip"  # clip, remove, impute


@dataclass
class ScalingConfig:
    """Cấu hình Scaling & Normalization"""
    method: str = "minmax"  # standard, minmax, robust
    features_to_scale: List[str] = None  # None = tất cả numeric


@dataclass
class ImputationConfig:
    """Cấu hình KNN Imputation"""
    method: str = "knn"  # knn, mean, forward_fill
    n_neighbors: int = 5
    weight_type: str = "distance"


@dataclass
class TransformConfig:
    """Cấu hình Log Transform"""
    enabled: bool = True
    features: List[str] = None  # None = tất cả positive cols
    base: float = 10
    add_constant: float = 1.0  # Để tránh log(0)

class L1NormalizationConfig:
    """Cấu hình L1 Normalization"""
    enabled: bool = False
    features: List[str] = None  # None = tất cả numeric

class ConfigHandler:
    """Quản lý cấu hình pipeline"""
    
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = {}
    
    def load_config(self, path: str) -> Dict[str, Any]:
        """Load file YAML từ ổ cứng """
        with open(path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
        return self.config
    
    def save_config(self, path: str, config: Dict[str, Any] = None):
        """Lưu YAML vào ổ cứng"""
        config_to_save = config or self.config
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
    
    def create_default_config(self) -> Dict[str, Any]:
        """Tạo cấu hình mặc định"""
        # Tính toán paths động từ project root (parent của pipeline_execution/)
        project_root = Path(__file__).parent.parent.parent
        input_folder = str(project_root / 'nam')
        output_folder = str(project_root / 'pipeline_execution' / 'outputs')
        
        return {
            'pipeline': {
                'name': 'WorldBank Data Pipeline',
                'version': '1.0',
                'input_folder': input_folder,
                'output_folder': output_folder,
            },
            'phase1': {
                'log_transform': asdict(TransformConfig()),
                'outlier_detection': asdict(OutlierConfig()),
            },
            'phase2': {
                'scaling': asdict(ScalingConfig()),
                'imputation': asdict(ImputationConfig()),
            },
            'diagnostics': {
                'distribution_bins': 30,
                'correlation_threshold': 0.8,
                'missing_threshold': 0.5,
            }
        }
    
    def get_value(self, *keys, default=None):
        """Lấy giá trị từ config bằng keys (nested dicts)"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default
    
    def validate_config(self) -> tuple[bool, List[str]]:
        """Kiểm tra tính hợp lệ của config"""
        errors = []
        
        # Kiểm tra các trường bắt buộc
        required_fields = [
            ('pipeline', 'input_folder'),
            ('pipeline', 'output_folder'),
        ]
        
        for *path, key in required_fields:
            if self.get_value(*path, key) is None:
                errors.append(f"Missing required field: {'.'.join(path + [key])}")
        
        # Kiểm tra outlier method
        outlier_method = self.get_value('phase1', 'outlier_detection', 'method')
        if outlier_method not in ['iqr', 'zscore', 'isolation_forest', 'lof', 'mahalanobis']:
            errors.append(f"Invalid outlier method: {outlier_method}")
        
        # Kiểm tra scaling method
        scaling_method = self.get_value('phase2', 'scaling', 'method')
        if scaling_method is not None and scaling_method not in ['standard', 'minmax', 'robust', 'iqr']:
            errors.append(f"Invalid scaling method: {scaling_method}")
        
        return len(errors) == 0, errors
    
    def print_config(self):
        """In cấu hình ra console"""
        print(json.dumps(self.config, indent=2, ensure_ascii=False))
