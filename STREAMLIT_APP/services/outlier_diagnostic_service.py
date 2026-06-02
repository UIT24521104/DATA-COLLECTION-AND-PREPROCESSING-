"""
Outlier Diagnostic Service - Analyze univariate and multivariate outliers by country.

Provides methods to:
1. Detect univariate outliers per feature and identify which countries are outliers
2. Detect multivariate outliers (interaction-based outliers)
3. Distinguish between signal (meaningful insights) and noise (measurement errors)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
import warnings

warnings.filterwarnings('ignore')


class OutlierDiagnostics:
    """Analyze outliers by country and feature for signal vs noise discrimination."""

    def __init__(self, data: pd.DataFrame, country_col: str = 'Country', year_col: str = 'Year'):
        """
        Initialize outlier diagnostics.
        
        Args:
            data: DataFrame containing the data (should be merged dataset with all features)
            country_col: Name of country column
            year_col: Name of year column
        """
        self.data = data.copy()
        self.country_col = country_col
        self.year_col = year_col
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove country and year from numeric columns if present
        if country_col in self.numeric_cols:
            self.numeric_cols.remove(country_col)
        if year_col in self.numeric_cols:
            self.numeric_cols.remove(year_col)

    def detect_univariate_outliers_iqr(self, threshold: float = 1.5) -> Dict[str, Dict]:
        """
        Detect univariate outliers using IQR method for each feature.
        
        Args:
            threshold: IQR multiplier (default 1.5). Use 3.0 for extreme outliers.
        
        Returns:
            Dictionary with feature names as keys, containing outlier information per country
        """
        results = {}
        
        for feature in self.numeric_cols:
            feature_data = self.data[[self.country_col, self.year_col, feature]].dropna()
            
            Q1 = feature_data[feature].quantile(0.25)
            Q3 = feature_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Identify outliers
            is_outlier = (feature_data[feature] < lower_bound) | (feature_data[feature] > upper_bound)
            outlier_rows = feature_data[is_outlier].copy()
            
            # Group by country
            outlier_by_country = {}
            if len(outlier_rows) > 0:
                for country in outlier_rows[self.country_col].unique():
                    country_outliers = outlier_rows[outlier_rows[self.country_col] == country]
                    outlier_by_country[country] = {
                        'count': len(country_outliers),
                        'values': country_outliers[feature].values.tolist(),
                        'years': country_outliers[self.year_col].values.tolist(),
                        'bounds': {'lower': lower_bound, 'upper': upper_bound},
                        'severity': 'extreme' if any(abs(country_outliers[feature]) > 3 * IQR) else 'moderate'
                    }
            
            results[feature] = {
                'outlier_countries': outlier_by_country,
                'total_outliers': len(outlier_rows),
                'total_points': len(feature_data),
                'outlier_pct': (len(outlier_rows) / len(feature_data) * 100) if len(feature_data) > 0 else 0,
                'statistics': {
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                }
            }
        
        return results

    def detect_univariate_outliers_zscore(self, threshold: float = 3.0) -> Dict[str, Dict]:
        """
        Detect univariate outliers using Z-score method (for normally distributed data).
        
        Args:
            threshold: Z-score threshold (default 3.0 for extreme outliers, 2.5 for high outliers)
        
        Returns:
            Dictionary with feature names as keys, containing outlier information per country
        """
        results = {}
        
        for feature in self.numeric_cols:
            feature_data = self.data[[self.country_col, self.year_col, feature]].dropna()
            
            if len(feature_data) < 2:
                continue
            
            z_scores = np.abs(zscore(feature_data[feature]))
            is_outlier = z_scores > threshold
            outlier_rows = feature_data[is_outlier].copy()
            
            # Group by country
            outlier_by_country = {}
            if len(outlier_rows) > 0:
                for country in outlier_rows[self.country_col].unique():
                    country_outliers = outlier_rows[outlier_rows[self.country_col] == country]
                    
                    # Get z-scores for this country's outliers
                    country_mask = feature_data[self.country_col] == country
                    country_z_scores = z_scores[country_mask]
                    country_is_outlier = is_outlier[country_mask]
                    country_z_scores_outliers = country_z_scores[country_is_outlier]
                    
                    outlier_by_country[country] = {
                        'count': len(country_outliers),
                        'values': country_outliers[feature].values.tolist(),
                        'years': country_outliers[self.year_col].values.tolist(),
                        'z_scores': country_z_scores_outliers.tolist(),
                        'severity': 'extreme' if any(country_z_scores_outliers > 4) else 'high'
                    }
            
            results[feature] = {
                'outlier_countries': outlier_by_country,
                'total_outliers': len(outlier_rows),
                'total_points': len(feature_data),
                'outlier_pct': (len(outlier_rows) / len(feature_data) * 100) if len(feature_data) > 0 else 0,
            }
        
        return results

    def detect_multivariate_outliers(self, method: str = 'isolation_forest') -> Dict:
        """
        Detect multivariate outliers (countries that are outliers across feature interactions).
        
        Args:
            method: 'isolation_forest' or 'mahalanobis'
        
        Returns:
            Dictionary with multivariate outlier information
        """
        # Prepare clean data
        data_clean = self.data[[self.country_col, self.year_col] + self.numeric_cols].dropna()
        
        if len(data_clean) < 2:
            return {'error': 'Insufficient data for multivariate analysis'}
        
        X = data_clean[self.numeric_cols].values
        
        if method == 'isolation_forest':
            # Isolation Forest for multivariate outliers
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            outlier_scores = iso_forest.score_samples(X)
            
            # Identify outlier points
            is_multivar_outlier = outlier_labels == -1
            
        elif method == 'mahalanobis':
            # Mahalanobis distance for multivariate outliers
            mean = np.mean(X, axis=0)
            cov = np.cov(X.T)
            try:
                inv_cov = np.linalg.inv(cov)
                mahal_dist = []
                for point in X:
                    md = mahalanobis(point, mean, inv_cov)
                    mahal_dist.append(md)
                
                # Use chi-squared threshold for outlier detection
                from scipy.stats import chi2
                threshold = chi2.ppf(0.95, df=len(self.numeric_cols))
                is_multivar_outlier = np.array(mahal_dist) > threshold
                outlier_scores = mahal_dist
            except np.linalg.LinAlgError:
                return {'error': 'Singular covariance matrix - insufficient data variation'}
        else:
            return {'error': f'Unknown method: {method}'}
        
        # Group multivariate outliers by country
        outlier_indices = np.where(is_multivar_outlier)[0]
        multivar_outlier_data = data_clean.iloc[outlier_indices].copy()
        multivar_outlier_data['anomaly_score'] = outlier_scores[outlier_indices]
        
        outlier_by_country = {}
        if len(multivar_outlier_data) > 0:
            for country in multivar_outlier_data[self.country_col].unique():
                country_data = multivar_outlier_data[
                    multivar_outlier_data[self.country_col] == country
                ]
                
                outlier_by_country[country] = {
                    'count': len(country_data),
                    'anomaly_scores': country_data['anomaly_score'].values.tolist(),
                    'years': country_data[self.year_col].values.tolist(),
                    'avg_score': float(country_data['anomaly_score'].mean()),
                    'max_score': float(country_data['anomaly_score'].max()),
                }
        
        return {
            'method': method,
            'total_multivar_outliers': int(is_multivar_outlier.sum()),
            'total_points': len(data_clean),
            'outlier_pct': float(is_multivar_outlier.sum() / len(data_clean) * 100),
            'outlier_countries': outlier_by_country,
            'threshold': float(threshold) if method == 'mahalanobis' else 'contamination=0.1',
        }

    def summarize_outliers_by_country(self, 
                                     univariate_results: Dict | None = None,
                                     multivariate_results: Dict | None = None) -> pd.DataFrame:
        """
        Create summary DataFrame showing which features have outliers per country.
        
        Args:
            univariate_results: Results from detect_univariate_outliers_iqr
            multivariate_results: Results from detect_multivariate_outliers
        
        Returns:
            DataFrame with country-level outlier summary
        """
        if univariate_results is None:
            univariate_results = self.detect_univariate_outliers_iqr()
        if multivariate_results is None:
            multivariate_results = self.detect_multivariate_outliers()
        
        # Collect all countries mentioned in outliers
        all_countries = set()
        for feature_info in univariate_results.values():
            all_countries.update(feature_info['outlier_countries'].keys())
        all_countries.update(multivariate_results.get('outlier_countries', {}).keys())
        
        # Build summary table
        summary_rows = []
        for country in sorted(all_countries):
            row = {'Country': country}
            
            # Univariate outliers
            univariate_features = []
            for feature, feature_info in univariate_results.items():
                if country in feature_info['outlier_countries']:
                    count = feature_info['outlier_countries'][country]['count']
                    univariate_features.append(f"{feature}({count})")
            
            row['Univariate_Outliers'] = ', '.join(univariate_features) if univariate_features else 'None'
            row['Univariate_Count'] = len(univariate_features)
            
            # Multivariate outliers
            if country in multivariate_results.get('outlier_countries', {}):
                row['Multivariate_Outliers'] = 'Yes'
                row['Multivariate_Count'] = multivariate_results['outlier_countries'][country]['count']
                row['Avg_Anomaly_Score'] = multivariate_results['outlier_countries'][country]['avg_score']
            else:
                row['Multivariate_Outliers'] = 'No'
                row['Multivariate_Count'] = 0
                row['Avg_Anomaly_Score'] = 0.0
            
            # Combined outlier risk
            total_outlier_features = row['Univariate_Count'] + row['Multivariate_Count']
            if total_outlier_features >= 3:
                row['Risk_Level'] = 'High'
            elif total_outlier_features >= 1:
                row['Risk_Level'] = 'Medium'
            else:
                row['Risk_Level'] = 'Low'
            
            summary_rows.append(row)
        
        return pd.DataFrame(summary_rows)

    def get_outlier_report(self, iqr_threshold: float = 1.5) -> Dict:
        """
        Generate comprehensive outlier diagnostic report.
        
        Returns:
            Dictionary containing all diagnostic information
        """
        univariate_iqr = self.detect_univariate_outliers_iqr(threshold=iqr_threshold)
        univariate_zscore = self.detect_univariate_outliers_zscore(threshold=3.0)
        multivariate = self.detect_multivariate_outliers(method='isolation_forest')
        summary = self.summarize_outliers_by_country(univariate_iqr, multivariate)
        
        return {
            'univariate_iqr': univariate_iqr,
            'univariate_zscore': univariate_zscore,
            'multivariate': multivariate,
            'summary': summary,
            'timestamp': pd.Timestamp.now().isoformat(),
        }

    def export_outlier_report(self, output_dir: Path, report: Dict | None = None) -> Dict[str, Path]:
        """
        Export outlier report to CSV files.
        
        Args:
            output_dir: Directory to save report files
            report: Report dictionary from get_outlier_report (generates if None)
        
        Returns:
            Dictionary with file paths
        """
        if report is None:
            report = self.get_outlier_report()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Summary by country
        summary_file = output_dir / 'outlier_summary_by_country.csv'
        report['summary'].to_csv(summary_file, index=False)
        files['summary'] = summary_file
        
        # Univariate outliers detail
        univariate_details = []
        for feature, feature_info in report['univariate_iqr'].items():
            for country, outlier_info in feature_info['outlier_countries'].items():
                for i, (value, year) in enumerate(zip(outlier_info['values'], outlier_info['years'])):
                    univariate_details.append({
                        'Feature': feature,
                        'Country': country,
                        'Year': year,
                        'Value': value,
                        'Lower_Bound': outlier_info['bounds']['lower'],
                        'Upper_Bound': outlier_info['bounds']['upper'],
                        'Severity': outlier_info['severity'],
                    })
        
        if univariate_details:
            univariate_file = output_dir / 'outliers_univariate_detail.csv'
            pd.DataFrame(univariate_details).to_csv(univariate_file, index=False)
            files['univariate'] = univariate_file
        
        # Multivariate outliers detail
        if report['multivariate'].get('outlier_countries'):
            multivariate_details = []
            for country, outlier_info in report['multivariate']['outlier_countries'].items():
                for i, (score, year) in enumerate(zip(outlier_info['anomaly_scores'], outlier_info['years'])):
                    multivariate_details.append({
                        'Country': country,
                        'Year': year,
                        'Anomaly_Score': score,
                        'Avg_Score': outlier_info['avg_score'],
                    })
            
            multivariate_file = output_dir / 'outliers_multivariate_detail.csv'
            pd.DataFrame(multivariate_details).to_csv(multivariate_file, index=False)
            files['multivariate'] = multivariate_file
        
        return files
