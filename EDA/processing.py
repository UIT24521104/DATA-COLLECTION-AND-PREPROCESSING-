from __future__ import annotations
"""
Module: Data Processing (Refactored)
Unified functions for data preprocessing:
- Transform: log_transform, quantile_transform, l1_normalization, l2_normalization
- Outlier Detection: IQR (univariate), LOF (multivariate), Mahalanobis, Isolation Forest
- Outlier Handling: clip/drop for univariate, drop for multivariate
- Missing Values: SimpleImputer (mean/median/most_frequent/constant), IterativeImputer (Tree-based)
- Scaling: Standard, MinMax, Robust, IQR-based
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # Enable experimental feature
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pickle


class DataProcessor:
    """Unified data processing with transform, outlier handling, imputation, scaling"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.metadata = {
            'original_shape': self.df.shape,
            'transformations': [],
        }
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Loại bỏ Country và Year nếu có
        self.numeric_cols = [col for col in self.numeric_cols if col not in ['Country', 'Year']]
        
        self.scaler = None
        self.imputer = None
    
    # ==================== TRANSFORM ====================
    def transform(self, transform_type: str = 'log', features: List[str] = None, 
                  add_constant: float = 1.0, n_quantiles: int = 10, 
                  output_distribution: str = 'normal', apply_l1: bool = False,
                  apply_l2: bool = False,
                  random_state: int = 42) -> pd.DataFrame:
        """
        Unified transform function: log, quantile, L1/L2 normalization
        """
        if transform_type is None:
            if apply_l1:
                self._apply_l1_normalization(features)
            if apply_l2:
                self._apply_l2_normalization(features)
            return self.df
        
        if transform_type == 'log':
            self._apply_log_transform(features, add_constant)
        elif transform_type == 'quantile':
            self._apply_quantile_transform(features, n_quantiles, output_distribution, random_state)
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")
        
        if apply_l1:
            self._apply_l1_normalization(features)
        if apply_l2:
            self._apply_l2_normalization(features)
        
        return self.df
    
    def _apply_log_transform(self, features: List[str] = None, add_constant: float = 1.0):
        """Apply log10 transform"""
        if features is None:
            features = [col for col in self.numeric_cols 
                       if (self.df[col].dropna() > 0).all()]
        
        for col in features:
            if col in self.df.columns:
                self.df[col] = np.log10(self.df[col] + add_constant)
        
        self.metadata['transformations'].append({
            'type': 'log_transform',
            'columns': features,
            'add_constant': add_constant
        })
    
    def _apply_quantile_transform(self, features: List[str] = None, n_quantiles: int = 10,
                                  output_distribution: str = 'normal', random_state: int = 42):
        """Apply quantile transform for skewed distributions"""
        if features is None:
            features = [col for col in self.numeric_cols if col in self.df.columns]
        
        valid_features = [col for col in features if col in self.df.columns]
        n_samples = len(self.df)
        actual_quantiles = min(n_quantiles, n_samples)
        
        transformer = QuantileTransformer(
            n_quantiles=actual_quantiles,
            output_distribution=output_distribution,
            random_state=random_state
        )
        
        self.df[valid_features] = transformer.fit_transform(self.df[valid_features])
        
        self.metadata['transformations'].append({
            'type': 'quantile_transform',
            'columns': valid_features,
            'output_distribution': output_distribution,
            'n_quantiles': actual_quantiles
        })
    
    def _apply_l1_normalization(self, features: List[str] = None):
        """Apply L1 normalization (probability distribution): each row sums to 1"""
        if features is None:
            features = [col for col in self.numeric_cols if col in self.df.columns]
        
        # Đảm bảo không có số âm
        self.df[features] = self.df[features].clip(lower=0)
        
        # Tính tổng theo hàng
        row_sums = self.df[features].sum(axis=1)
        row_sums = row_sums.replace(0, 1e-9)
        
        # L1 normalization
        self.df[features] = self.df[features].div(row_sums, axis=0)
        
        self.metadata['transformations'].append({
            'type': 'l1_normalization',
            'columns': features
        })

    def _apply_l2_normalization(self, features: List[str] = None):
        """Apply L2 normalization: each row has unit Euclidean norm"""
        if features is None:
            features = [col for col in self.numeric_cols if col in self.df.columns]

        valid_features = [col for col in features if col in self.df.columns]
        if not valid_features:
            return

        row_norms = np.sqrt((self.df[valid_features] ** 2).sum(axis=1))
        row_norms = row_norms.replace(0, 1e-9)

        self.df[valid_features] = self.df[valid_features].div(row_norms, axis=0)

        self.metadata['transformations'].append({
            'type': 'l2_normalization',
            'columns': valid_features
        })
    
    # ==================== OUTLIER DETECTION ====================
    def outlier_detection(self, method: str = 'iqr', features: List[str] = None,
                          iqr_multiplier: float = 1.5, 
                          lof_params: Dict[str, Any] = None,
                          mahalanobis_params: Dict[str, Any] = None,
                          isolation_forest_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect outliers using univariate (IQR) or multivariate methods
        """
        if method is None:
            return {"n_outliers": 0}
        
        if features is None:
            features = self.numeric_cols
        
        valid_features = [col for col in features if col in self.df.columns]
        
        if method == 'iqr':
            return self._detect_outliers_iqr(valid_features, iqr_multiplier)
        elif method == 'lof':
            return self._detect_outliers_lof(valid_features, lof_params or {})
        elif method == 'mahalanobis':
            return self._detect_outliers_mahalanobis(valid_features, mahalanobis_params or {})
        elif method == 'isolation_forest':
            return self._detect_outliers_isolation_forest(valid_features, isolation_forest_params or {})
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_outliers_iqr(self, features: List[str], multiplier: float = 1.5) -> Dict[str, Any]:
        """Detect univariate outliers using IQR"""
        outlier_info = {'method': 'iqr', 'outliers_by_feature': {}}
        
        for col in features:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_indices = self.df[outlier_mask].index.tolist()
            
            if len(outlier_indices) > 0:
                outlier_info['outliers_by_feature'][col] = {
                    'indices': outlier_indices,
                    'count': len(outlier_indices),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return outlier_info
    
    def _detect_outliers_lof(self, features: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect multivariate outliers using Local Outlier Factor"""
        n_neighbors = params.get('n_neighbors', 20)
        contamination = params.get('contamination', 0.05)
        metric = params.get('metric', 'euclidean')
        
        valid_df = self.df[features].dropna()
        
        if len(valid_df) < n_neighbors + 1:
            return {'method': 'lof', 'outlier_indices': []}
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric
        )
        
        outlier_predictions = lof.fit_predict(valid_df)
        outlier_indices = valid_df[outlier_predictions == -1].index.tolist()
        
        return {
            'method': 'lof',
            'outlier_indices': outlier_indices,
            'n_outliers': len(outlier_indices),
            'params': {'n_neighbors': n_neighbors, 'contamination': contamination, 'metric': metric}
        }

    def _detect_outliers_mahalanobis(self, features: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect multivariate outliers using Mahalanobis distance"""
        contamination = params.get('contamination', 0.05)
        threshold_quantile = params.get('threshold_quantile', None)

        valid_df = self.df[features].dropna()

        if len(valid_df) < 3 or len(features) == 0:
            return {'method': 'mahalanobis', 'outlier_indices': []}

        if threshold_quantile is None:
            threshold_quantile = 1 - contamination

        estimator = LedoitWolf()
        estimator.fit(valid_df)
        distances = estimator.mahalanobis(valid_df)
        threshold = np.quantile(distances, threshold_quantile)
        outlier_indices = valid_df[distances > threshold].index.tolist()

        return {
            'method': 'mahalanobis',
            'outlier_indices': outlier_indices,
            'n_outliers': len(outlier_indices),
            'params': {
                'contamination': contamination,
                'threshold_quantile': threshold_quantile
            },
            'threshold': threshold,
            'distances': distances.tolist()
        }

    def _detect_outliers_isolation_forest(self, features: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect multivariate outliers using Isolation Forest"""
        n_estimators = params.get('n_estimators', 100)
        contamination = params.get('contamination', 0.05)
        random_state = params.get('random_state', 42)
        max_samples = params.get('max_samples', 'auto')

        valid_df = self.df[features].dropna()

        if len(valid_df) < 3:
            return {'method': 'isolation_forest', 'outlier_indices': []}

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            max_samples=max_samples
        )

        predictions = model.fit_predict(valid_df)
        outlier_indices = valid_df[predictions == -1].index.tolist()

        return {
            'method': 'isolation_forest',
            'outlier_indices': outlier_indices,
            'n_outliers': len(outlier_indices),
            'params': {
                'n_estimators': n_estimators,
                'contamination': contamination,
                'random_state': random_state,
                'max_samples': max_samples
            }
        }
    
    # ==================== OUTLIER HANDLING ====================
    def handle_outliers(self, outlier_info: Dict[str, Any], action: str = 'clip', 
                        features: List[str] = None, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Handle detected outliers
        """
        method = outlier_info.get('method', 'iqr')
        
        if features is None:
            features = self.numeric_cols
        
        if method == 'iqr':
            self._handle_outliers_iqr(features, action, iqr_multiplier)
        elif method in ['lof', 'mahalanobis', 'isolation_forest']:
            self._handle_outliers_lof(outlier_info.get('outlier_indices', []), action)
        
        self.metadata['transformations'].append({
            'type': 'outlier_handling',
            'method': method,
            'action': action
        })
        
        return self.df
    
    def _handle_outliers_iqr(self, features: List[str], action: str = 'clip', multiplier: float = 1.5):
        """Handle IQR outliers"""
        for col in features:
            if col not in self.df.columns:
                continue
            
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                if action == 'clip':
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == 'drop':
                    self.df = self.df[~outlier_mask]
                elif action == 'impute':
                    self.df.loc[outlier_mask, col] = np.nan
    
    def _handle_outliers_lof(self, outlier_indices: List[int], action: str = 'drop'):
        """Handle LOF multivariate outliers"""
        if action == 'drop':
            self.df = self.df.drop(index=outlier_indices, errors='ignore')
        elif action == 'impute':
            self.df.loc[outlier_indices, self.numeric_cols] = np.nan
    
    # ==================== MISSING VALUES HANDLING ====================
    def handle_missing_values(self, method: str = 'univariate', strategy: str = 'mean',
                              features: List[str] = None, 
                              tree_model: str = 'random_forest',
                              drop_threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values
        """
        if features is None:
            features = [col for col in self.numeric_cols if col in self.df.columns]
        
        valid_features = [col for col in features if col in self.df.columns]
        
        # Drop columns with too many missing values
        self._drop_high_missing_columns(valid_features, drop_threshold)
        
        # Impute remaining missing values
        if method == 'univariate':
            self._impute_univariate(valid_features, strategy)
        elif method == 'multivariate':
            self._impute_multivariate(valid_features, tree_model)
        elif method == 'drop':
            self.df = self.df.dropna(subset=valid_features)
        
        self.metadata['transformations'].append({
            'type': 'missing_values_handling',
            'method': method,
            'strategy': strategy
        })
        
        return self.df
    
    def _drop_high_missing_columns(self, features: List[str], threshold: float = 0.5):
        """Drop columns with >threshold% missing"""
        cols_to_drop = []
        for col in features:
            missing_pct = self.df[col].isna().sum() / len(self.df)
            if missing_pct > threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
    
    def _impute_univariate(self, features: List[str], strategy: str = 'mean'):
        """SimpleImputer for univariate imputation"""
        valid_features = [col for col in features if self.df[col].isna().any()]
        
        if not valid_features:
            return
        
        if strategy == 'constant':
            imputer = SimpleImputer(strategy=strategy, fill_value=0)
        else:
            imputer = SimpleImputer(strategy=strategy)
        
        self.df[valid_features] = imputer.fit_transform(self.df[valid_features])
    
    def _impute_multivariate(self, features: List[str], tree_model: str = 'random_forest'):
        """IterativeImputer with tree-based estimator"""
        valid_features = [col for col in features if self.df[col].isna().any()]
        
        if not valid_features:
            return
        
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        self.imputer = IterativeImputer(estimator=estimator, max_iter=10, random_state=42, verbose=2 )
        
        self.df[valid_features] = self.imputer.fit_transform(self.df[valid_features])
    
    # ==================== SCALING ====================
    def scaling(self, method: str = 'minmax', features: List[str] = None) -> Tuple[pd.DataFrame, Any]:
        """
        Scale/normalize features
        """
        if method is None:
            return self.df, None
        
        if features is None:
            features = [col for col in self.numeric_cols if col in self.df.columns]
        
        valid_features = [col for col in features if col in self.df.columns]
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'iqr':
            self.scaler = IQRScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        self.df[valid_features] = self.scaler.fit_transform(self.df[valid_features])
        
        self.metadata['transformations'].append({
            'type': 'scaling',
            'method': method,
            'columns': valid_features
        })
        
        return self.df, self.scaler
    
    # ==================== MODEL PERSISTENCE ====================
    def save_scaler(self, output_path: str):
        """Save scaler model"""
        if self.scaler is None:
            raise ValueError("Scaler is None, run scaling first")
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def save_imputer(self, output_path: str):
        """Save imputer model"""
        if self.imputer is None:
            return
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.imputer, f)
    
    def load_scaler(self, input_path: str):
        """Load scaler model for inference"""
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Scaler file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def load_imputer(self, input_path: str):
        """Load imputer model for inference"""
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Imputer file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            self.imputer = pickle.load(f)
    
    # ==================== PIPELINE ORCHESTRATION ====================
    def run_processing_pipeline(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete preprocessing pipeline based on config
        """
        # Step 1: Transform
        if config.get('transform', {}).get('enabled', False):
            transform_config = config['transform']
            self.transform(
                transform_type=transform_config.get('type', 'log'),
                add_constant=transform_config.get('add_constant', 1.0),
                n_quantiles=transform_config.get('n_quantiles', 10),
                output_distribution=transform_config.get('output_distribution', 'normal'),
                apply_l1=transform_config.get('apply_l1', False),
                apply_l2=transform_config.get('apply_l2', False)
            )
        
        # Step 2: Outlier Detection
        outlier_detection_config = config.get('outlier_detection', {})
        if outlier_detection_config.get('enabled', True):
            method = outlier_detection_config.get('method', 'iqr')
            outlier_info = self.outlier_detection(
                method=method,
                iqr_multiplier=outlier_detection_config.get('iqr_multiplier', 1.5),
                lof_params=outlier_detection_config.get('lof_params', {}),
                mahalanobis_params=outlier_detection_config.get('mahalanobis_params', {}),
                isolation_forest_params=outlier_detection_config.get('isolation_forest_params', {})
            )
            
            # Step 3: Handle Outliers
            outlier_handling_config = config.get('outlier_handling', {})
            if outlier_handling_config.get('enabled', True):
                self.handle_outliers(
                    outlier_info=outlier_info,
                    action=outlier_handling_config.get('action', 'clip'),
                    iqr_multiplier=outlier_detection_config.get('iqr_multiplier', 1.5)
                )
        
        # Step 4: Missing Values
        missing_config = config.get('missing_values', {})
        if missing_config.get('enabled', True):
            self.handle_missing_values(
                method=missing_config.get('method', 'univariate'),
                strategy=missing_config.get('strategy', 'mean'),
                drop_threshold=missing_config.get('drop_threshold', 0.5)
            )
        
        # Step 5: Scaling
        scaling_config = config.get('scaling', {})
        if scaling_config.get('enabled', True):
            self.scaling(
                method=scaling_config.get('method', 'minmax')
            )
        
        return self.df, {'scaler': self.scaler, 'imputer': self.imputer}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about transformations"""
        return self.metadata


# ==================== CUSTOM SCALERS ====================
class IQRScaler:
    """Custom IQR-based scaler: (X - Q1) / (Q3 - Q1) → scales to approximately 0-1"""
    
    def __init__(self):
        self.q1 = None
        self.q3 = None
    
    def fit(self, X):
        """Fit scaler"""
        self.q1 = X.quantile(0.25)
        self.q3 = X.quantile(0.75)
        return self
    
    def transform(self, X):
        """Transform data"""
        return (X - self.q1) / (self.q3 - self.q1)
    
    def fit_transform(self, X):
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)