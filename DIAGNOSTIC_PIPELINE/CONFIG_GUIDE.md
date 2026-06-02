# Hướng Dẫn Sử Dụng Config.yaml

## I. Tổng Quan

- Giải thích vai trò của file config.yaml
- Cấu trúc của pipelien có 2 phase: phase 0 và phase 1
- Config controls tất cả preprocessing steps

## II. Cấu Trúc File Config

Xử lý dữ liệu thiếu: univariate(Impute) và multivariate(KNN)
Xử lý outliers: outlier đơn biến , outlier đa biến
Xử lý dữ liệu lệch (transform):log_tranform và quantile_transform
Scaling

## III. Chi tiết từng option

#### 1. Quantile Transform

transform(transform_type, apply_l1): 'log' , 'quantile', 'l1'
outlier detection: IQR, LOF
Handle outliers: Univariate, Multivariate
Missing values: Univaritate (mean, median, most_frequent, constant) và Multivariate (RandomForestRegressor)
scaling: minmax, robust, iqr
