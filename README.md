# 🌍 WorldBank Data Analysis Project

**Dự báo tăng trưởng GDP của Việt Nam 2024-2025 bằng cách so sánh với các nền kinh tế tương đồng**

---

## 📖 Mục Lục

1. [🚀 Quick Start](#-quick-start) - Bắt đầu nhanh trong 5 phút
2. [📁 Project Structure](#-project-structure) - Cấu trúc dự án
3. [🔧 Installation](#-installation) - Cài đặt môi trường
4. [📊 Dashboard Guide](#-dashboard-guide) - Hướng dẫn 2 Streamlit App
5. [🔄 Pipeline Workflow](#-pipeline-workflow) - Quy trình xử lý dữ liệu
6. [📈 Outputs](#-outputs) - File đầu ra
7. [🐛 Troubleshooting](#-troubleshooting) - Xử lý sự cố

---

## 🚀 Quick Start

### Yêu cầu

- Python 3.9+
- Git
- ~2GB dung lượng ổ cứng

### 1️⃣ Clone hoặc tải project

```bash
cd e:\WorldBank-Data-Analysis-Project
```

### 2️⃣ Tạo Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Hoặc cmd.exe
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Cài đặt Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4️⃣ Chạy Pipeline (tạo dữ liệu)

```bash
cd DIAGNOSTIC_PIPELINE
python run_pipeline.py full --scenario default
```

⏱️ **Thời gian**: ~10 phút. Kết quả được lưu vào `DIAGNOSTIC_PIPELINE/outputs/latest/`

### 5️⃣ Chạy Streamlit Dashboards

Mở 2 terminal khác nhau:

**Terminal 1 - Main Dashboard:**

```bash
cd STREAMLIT_APP
streamlit run app.py
```

**Terminal 2 - Trajectory Analysis:**

```bash
cd TRAJECTORY_APP
streamlit run app.py
```

✅ Mặc định mở tại:

- Main Dashboard: `http://localhost:8501`
- Trajectory App: `http://localhost:8502` (nếu tự động đổi port)

---

## 📁 Project Structure

```
WorldBank-Data-Analysis-Project/
│
├── 📄 README.md                                ← You are here
├── 📄 requirements.txt                         ← Python packages
├── 📄 run_full_pipeline.py                     ← Master script
│
├── 📂 RAW_DATASET/                              ← World Bank dữ liệu thô
│   ├── Access to electricity.csv
│   ├── GDP per capita.csv
│   ├── Individuals using the Internet.csv
│   └── ... (9 indicators)
│
├── 📂 DIAGNOSTIC_PIPELINE/                      ← Pipeline & Diagnostics (Phase 0-1)
│   ├── 📄 run_pipeline.py                      Main pipeline runner
│   ├── 📄 README.md                            Detailed pipeline guide
│   ├── 📄 test_integration.py
│   ├── 📁 modules/
│   │   ├── data_integration.py                (Phase 0: Data Merge)
│   │   ├── diagnostics.py                     (Phase 1: Auto Profiling)
│   │   ├── config_handler.py
│   │   ├── run_manager.py
│   │   ├── logger_setup.py
│   │   └── transformation_viz.py
│   └── 📁 outputs/latest/                      (Latest run results)
│       ├── config.yaml
│       ├── dataset_merged.csv                  Merged data
│       ├── diagnostic_report.csv               Diagnostics summary
│       ├── run_summary.json
│       └── logs/                               Pipeline logs
│
├── 📂 EDA/                                      ← Exploratory Data Analysis
│   ├── 📔 impute_missing_data.ipynb            (Jupyter notebook)
│   ├── 📄 processing.py                        (Data processing utilities)
│   ├── 📄 EDA_REPORT.md
│   └── 📁 visualizations_by_feature/
│
├── 📂 MODELS/                                   ← ML Models
│   ├── 📄 K-Means.py                           (Clustering)
│   ├── 📄 Trajectory.py                        (Similarity ranking)
│   ├── 📄 requirements.txt                     (Model dependencies)
│   └── 📁 plots/                               (Output visualizations)
│
├── 📂 STREAMLIT_APP/                            ← 📊 Main Dashboard
│   ├── 📄 app.py                               (Entry point)
│   ├── 📁 pages/
│   │   ├── 01_Data_Explorer.py
│   │   ├── 02_EDA_Interactive.py
│   │   ├── 03_Diagnostics.py
│   │   └── 06_Outlier_Diagnostics.py
│   ├── 📁 components/
│   │   ├── charts.py
│   │   ├── filters.py
│   │   └── metrics.py
│   ├── 📁 services/
│   │   ├── data_loader.py
│   │   ├── eda_service.py
│   │   ├── diagnostics_service.py
│   │   └── outlier_diagnostic_service.py
│   ├── 📁 utils/
│   │   ├── cache.py
│   │   ├── ui.py
│   ├── 📁 assets/
│   │   └── styles.css
│   └── 📄 __init__.py
│
├── 📂 TRAJECTORY_APP/                           ← 🇻🇳 Trajectory Analysis
│   └── 📄 app.py                               (Similarity ranking dashboard)
│
├── 📂 METADATA/                                 ← Schema & metadata
│   ├── metadata_before_preprocessing.csv
│   └── metadata_after_preprocessing.csv
│
├── 📂 DATA_AFTER_PREPROCESSING/                 ← Processed data
│   ├── data_final_merged.csv
│   └── dataset_merged.csv
│
└── 📂 .venv/                                    ← Virtual environment
```

---

## 🔧 Installation

### Step 1: Setup Python Environment

```bash
# Verify Python version (3.9+)
python --version

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate
```

### Step 2: Install Requirements

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

### Step 3: Test Installation

```bash
# Test Python imports
python -c "import pandas; import streamlit; import plotly; print('✅ All packages imported successfully')"

# Test Jupyter
jupyter --version

# Test Streamlit
streamlit --version
```

---

## 📊 Dashboard Guide

### Dashboard 1: Main Interactive Dashboard (`STREAMLIT_APP`)

**Mục đích**: Khám phá dữ liệu, phân tích EDA, và kiểm tra chất lượng dữ liệu

**Chạy:**

```bash
cd STREAMLIT_APP
streamlit run app.py
```

✅ Dữ liệu được tự động tải từ:

- `DATA_AFTER_PREPROCESSING/dataset_merged.csv` (ưu tiên)
- `DIAGNOSTIC_PIPELINE/outputs/latest/dataset_merged.csv` (fallback)

**Các trang:**

1. **Main Page** - Tổng quan dữ liệu (số rows, cols, missing values)
2. **01_Data_Explorer** - Lọc và khám phá dữ liệu theo quốc gia, năm
3. **02_EDA_Interactive** - Biểu đồ phân phối, tương quan
4. **03_Diagnostics** - Báo cáo chẩn đoán dữ liệu
5. **06_Outlier_Diagnostics** - Phát hiện ngoại lệ

**Features:**

- ✅ Filter theo Country, Year, Feature
- ✅ Data preview (head 50 rows)
- ✅ Missing value summary
- ✅ Distribution plots
- ✅ Correlation heatmap
- ✅ Outlier detection

---

### Dashboard 2: Trajectory Analysis (`TRAJECTORY_APP`)

**Mục đích**: So sánh quỹ đạo kinh tế Việt Nam với các nước tương đồng

**Chạy:**

```bash
cd TRAJECTORY_APP
streamlit run app.py
```

**Các chức năng:**

1. **Country Ranking** - Top 10 quốc gia tương đồng với VNM
2. **Trajectory Visualization** - Đồ thị so sánh các chỉ số kinh tế
3. **Similarity Metrics** - Vector similarity vs JSD distance
4. **Economic Indicators** - So sánh các chỉ số: FDI, Internet, GDP/capita,...

**Chỉ số sử dụng:**

- 🌾 Agriculture GDP (% GDP)
- 💡 Electricity Access (%)
- 🏢 Industry GDP (%)
- 🌐 Internet Usage (%)
- 💵 GDP per Capita (2015 USD)
- 🇻🇳 FDI (% GDP)
- 📊 Services GDP (%)

**Phương pháp so sánh:**

- **Vector Similarity**: Cosine/Euclidean/DTW distance
- **JSD (Jensen-Shannon Divergence)**: Xác suất phân phối
- **Combined Score**: Trung bình 2 phương pháp

---

## 🔄 Pipeline Workflow

### 3 Giai Đoạn Chính

```
RAW_DATASET (9 indicators)
    ↓
PHASE 0: DATA INTEGRATION
  • Merge 9 CSV từ World Bank
  • Lọc bỏ non-country territories
  • Kết hợp Country × Year
  → Output: dataset_merged.csv (4,110 rows)
    ↓
PHASE 1: AUTO PROFILING & DIAGNOSTICS
  • Phân tích missing values
  • Phát hiện outliers (IQR)
  • Tạo visualizations (7 PNG files)
  • Generate config recommendations
  → Output: diagnostic_report.csv, config.yaml
    ↓
USE DATA FOR YOUR ANALYSIS
  ✅ dataset_merged.csv ready for Streamlit dashboards
  ✅ Visualizations & reports in outputs/latest/
  ✅ Logs & metadata for debugging
```

### Chạy Pipeline

**Full pipeline (Phase 0 + 1):**

```bash
cd DIAGNOSTIC_PIPELINE
python run_pipeline.py full --scenario default
```

**Phase 0 only (merge data):**

```bash
cd DIAGNOSTIC_PIPELINE
python run_pipeline.py phase0 --scenario default
```

**Phase 1 only (diagnostics):**

```bash
cd DIAGNOSTIC_PIPELINE
python run_pipeline.py phase1 --scenario default --input ./outputs/latest/dataset_merged.csv
```

**Output được lưu vào:**

```
DIAGNOSTIC_PIPELINE/outputs/
├── latest/                          ← Latest results
│   ├── config.yaml
│   ├── dataset_merged.csv           ← Use this for Streamlit!
│   ├── diagnostic_report.csv
│   ├── run_summary.json
│   ├── boxplots.png
│   ├── histograms.png
│   ├── correlation_heatmap.png
│   └── ... (7 visualization files)
├── runs/
│   ├── run_20260602_162514_default/
│   ├── run_20260602_162501_test/
│   └── ...
└── scenarios/
```

---

## 📈 Outputs

### Phase 0 - Data Integration

- `dataset_merged.csv` - Merged data (Country, Year, 9 economic indicators)
- File size: ~500 KB
- Rows: 4,110 (265 countries, 21 years 2004-2024)

### Phase 1 - Diagnostics & Visualizations

| File                                   | Purpose                                          |
| -------------------------------------- | ------------------------------------------------ |
| `diagnostic_report.csv`                | Missing %, outlier count, statistics per feature |
| `config.yaml`                          | Auto-generated preprocessing recommendations     |
| `boxplots.png`                         | Distribution + outliers per feature              |
| `histograms.png`                       | Frequency distribution                           |
| `correlation_heatmap.png`              | Feature correlation matrix                       |
| `pairwise_scatter.png`                 | Pairwise scatter plots                           |
| `missing_values_by_year.png`           | Missing data trend over time                     |
| `matrix_plot.png` / `heatmap_plot.png` | Missing data patterns (missingno library)        |

### Use in Streamlit Apps

The merged dataset (`dataset_merged.csv`) is automatically loaded by:

- `STREAMLIT_APP/app.py` - Main dashboard
- `TRAJECTORY_APP/app.py` - Trajectory analysis

---

## 🐛 Troubleshooting

| Problem                                      | Solution                                                                     |
| -------------------------------------------- | ---------------------------------------------------------------------------- |
| `FileNotFoundError: RAW_DATASET not found`   | Ensure RAW_DATASET/ folder exists with 9 CSV files                           |
| `ModuleNotFoundError: No module named 'X'`   | Install dependencies: `pip install -r requirements.txt`                      |
| Pipeline exits with no output                | Check logs in `DIAGNOSTIC_PIPELINE/outputs/runs/run_TIMESTAMP/logs/`         |
| Phase 1 fails (dataset_merged.csv not found) | Run Phase 0 first: `cd DIAGNOSTIC_PIPELINE && python run_pipeline.py phase0` |
| `UnicodeDecodeError` or encoding issues      | Already fixed in code - should not occur                                     |
| Port 8501 already in use                     | Run dashboard on different port: `streamlit run app.py --server.port 8502`   |
| Streamlit can't find dataset_merged.csv      | Run pipeline in DIAGNOSTIC_PIPELINE folder first                             |

### Common Workflow Issues

**Issue**: "I ran the pipeline but Streamlit still shows no data"

**Solution**:

1. Verify pipeline completed successfully (check exit code = 0)
2. Verify file exists: `DIAGNOSTIC_PIPELINE/outputs/latest/dataset_merged.csv`
3. Streamlit automatically looks in the correct location
4. Restart Streamlit after pipeline completes

**Issue**: "Phase 1 won't run after Phase 0"

**Solution**: Option to specify Phase 0 output:

```bash
cd DIAGNOSTIC_PIPELINE
python run_pipeline.py phase1 --scenario mytest --input ./outputs/latest/dataset_merged.csv
```

---

## 📝 Notebook Files

Các Jupyter notebooks trong project:

### `EDA/impute_missing_data.ipynb`

**Mục đích**: Khám phá các phương pháp imputation

**Cách chạy:**

```bash
# Khởi động Jupyter Lab
jupyter lab

# Hoặc Jupyter Notebook
jupyter notebook

# Mở file: EDA/impute_missing_data.ipynb
```

**Cấu hình kernel:**

1. Vào Kernel → Select Kernel
2. Chọn `.venv` environment
3. Chạy các cell theo thứ tự

---

## 🔗 Liên kết Hữu Ích

- **World Bank Open Data**: https://data.worldbank.org
- **Streamlit Docs**: https://docs.streamlit.io
- **Jupyter Docs**: https://jupyter.org/install
- **Pandas Documentation**: https://pandas.pydata.org
- **Scikit-learn**: https://scikit-learn.org

---

## 👤 Author & Maintenance

**Project Start**: June 2026  
**Last Updated**: June 2, 2026

---

## 📋 Checklist - Before Running

- [ ] Python 3.9+ installed
- [ ] Virtual environment created & activated
- [ ] All requirements installed: `pip install -r requirements.txt`
- [ ] RAW_DATASET folder có 9 CSV files từ World Bank
- [ ] Đủ dung lượng ổ cứng (~2GB)
- [ ] Port 8501, 8502 không bị sử dụng

---

## 🎯 Next Steps

1. **Setup**: Follow [Installation](#-installation)
2. **Data**: Run `python run_full_pipeline.py default`
3. **Explore**: Open `STREAMLIT_APP` dashboard
4. **Analyze**: Open `TRAJECTORY_APP` for country comparison
5. **Export**: Save reports từ outputs/latest/

---

**Happy analyzing! 🚀** 📊
