# 🔍 Diagnostic Pipeline - Data Integration & Profiling

## Complete Multi-Phase Data Integration & Auto-Profiling Workflow

This pipeline provides **2 main phases**:

- **Phase 0**: Data Integration - Merge all World Bank CSV files
- **Phase 1**: Auto Profiling & Diagnostics - Analyze data quality, create reports & visualizations

**Note**: Phase 2 (preprocessing) is not available. Use the merged dataset from Phase 1 for custom preprocessing.

---

## 📋 Quick Start

### Prerequisites

```bash
Python 3.9+
pandas, numpy, scikit-learn, matplotlib, seaborn, pyyaml
```

### Installation

```bash
# From project root
cd DIAGNOSTIC_PIPELINE

# Install dependencies (if not already installed)
pip install -r ../requirements.txt
```

### Run Pipeline

**Full pipeline (Phase 0 + 1):**

```bash
python run_pipeline.py full --scenario default
```

**Backward-compatible syntax (same as above):**

```bash
python run_pipeline.py default
```

**Phase 0 only (data merge):**

```bash
python run_pipeline.py phase0 --scenario default
```

**Phase 1 only (diagnostics on existing merged data):**

```bash
python run_pipeline.py phase1 --scenario default --input ./outputs/latest/dataset_merged.csv
```

---

## 🔗 Integration with Streamlit Apps

The output `dataset_merged.csv` is automatically used by:

- `../STREAMLIT_APP/app.py` - Main interactive dashboard
- `../TRAJECTORY_APP/app.py` - Trajectory similarity analysis

The Streamlit apps load data from multiple locations (in priority order):

1. `DATA_AFTER_PREPROCESSING/dataset_merged.csv`
2. `DIAGNOSTIC_PIPELINE/outputs/latest/dataset_merged.csv`

---

## 📁 Folder Structure

```
DIAGNOSTIC_PIPELINE/
├── README.md                          (This file)
├── CONFIG_GUIDE.md                    Configuration guide
├── run_pipeline.py                    Main pipeline runner
├── test_integration.py                Integration tests
├── feature_encoding_rules.json         Feature mapping
│
├── modules/
│   ├── __init__.py
│   ├── data_integration.py            Phase 0: Data merging
│   ├── diagnostics.py                 Phase 1: Analysis & profiling
│   ├── config_handler.py              YAML configuration
│   ├── run_manager.py                 Output management
│   ├── logger_setup.py                Logging setup
│   └── transformation_viz.py          Visualization utilities
│
└── outputs/
    ├── latest/                        Latest run results
    │   ├── dataset_merged.csv         Merged data (Phase 0 output)
    │   ├── diagnostic_report.csv      Diagnostics summary
    │   ├── config.yaml                Auto-generated config
    │   ├── run_summary.json           Run metadata
    │   └── logs/                      Pipeline logs
    │
    ├── runs/                          All previous runs
    │   ├── run_20260602_162514_test/
    │   ├── run_20260602_162501_test/
    │   └── ...
    │
    └── scenarios/                     Scenario configs (optional)
```

---

## 🔄 Pipeline Workflow

```
INPUT: RAW_DATASET/
  (9 World Bank CSV files)
       ↓
PHASE 0: DATA INTEGRATION
  └─ Merge all CSVs → dataset_merged.csv
  └─ Filter non-country territories
  └─ Output: 4,110 rows × 11 columns (after filtering)
       ↓
PHASE 1: AUTO PROFILING & DIAGNOSTICS
  └─ Analyze distributions, missing values
  └─ Detect outliers (IQR method)
  └─ Generate visualizations:
     • boxplots.png
     • histograms.png
     • correlation_heatmap.png
     • pairwise_scatter.png
     • missing_values_by_year.png
     • matrix_plot.png (missingno)
     • heatmap_plot.png (missingno)
  └─ Output: diagnostic_report.csv, config.yaml
       ↓
OUTPUT: outputs/latest/
```

---

## 📊 Output Files

### Phase 0 Output

| File                 | Content                                            |
| -------------------- | -------------------------------------------------- |
| `dataset_merged.csv` | Merged data (Country, Year, 9 economic indicators) |
| `run_summary.json`   | Execution metadata                                 |

### Phase 1 Output

| File                                   | Content                                               |
| -------------------------------------- | ----------------------------------------------------- |
| `diagnostic_report.csv`                | Missing values %, outliers, distributions per feature |
| `boxplots.png`                         | Distribution by feature (outlier visualization)       |
| `histograms.png`                       | Frequency distribution                                |
| `correlation_heatmap.png`              | Feature correlation matrix                            |
| `pairwise_scatter.png`                 | Pairwise relationships                                |
| `missing_values_by_year.png`           | Missing data trend                                    |
| `matrix_plot.png` / `heatmap_plot.png` | Missing data patterns                                 |
| `config.yaml`                          | Recommendations for preprocessing                     |

---

## 🎯 Example Commands

```bash
# Run everything with default settings
python run_pipeline.py full

# Run with custom scenario name
python run_pipeline.py full --scenario my_test

# Just merge the data
python run_pipeline.py phase0 --scenario merge_only

# Only analyze data (requires phase0 output)
python run_pipeline.py phase1 --scenario analyze_only

# Phase 1 with custom input file
python run_pipeline.py phase1 --input path/to/dataset_merged.csv --scenario custom
```

---

## 📈 Data Overview

**Input**: 9 World Bank indicators

- Electricity access (%)
- Agriculture GDP (% of total)
- Domestic credit (% of GDP)
- Employment in services (%)
- FDI inflows (% of GDP)
- GDP per capita (2015 USD)
- Internet usage (%)
- Industry GDP (%)
- Services GDP (%)

**Time period**: 2004-2024 (21 years)  
**Countries**: ~265 countries + regions → ~200 countries (after filtering)  
**Total records**: 5,546 → 4,110 (after non-country filter)

---

## 🐛 Troubleshooting

| Problem                                    | Solution                                                             |
| ------------------------------------------ | -------------------------------------------------------------------- |
| `FileNotFoundError: RAW_DATASET not found` | Ensure `RAW_DATASET/` folder exists in project root with 9 CSV files |
| `ModuleNotFoundError`                      | Install requirements: `pip install -r ../requirements.txt`           |
| `No output in outputs/latest/`             | Check `outputs/runs/` for timestamped run directories                |
| Phase 1 fails to find dataset_merged.csv   | Run Phase 0 first, or specify `--input` path explicitly              |

---

## 🔧 Configuration

Edit `feature_encoding_rules.json` to customize feature column mappings.

Example:

```json
{
  "Access to electricity (% of population).csv": "Elec_Access",
  "GDP per capita (constant 2015 US$).csv": "GDPC_2015"
}
```

---

## 📝 Logs

All pipeline logs are saved to:

```
outputs/runs/run_TIMESTAMP_SCENARIO/logs/
  ├── pipeline.log         Main pipeline log
  ├── data_integration.log Phase 0 log
  └── diagnostics.log      Phase 1 log
```

Check logs for detailed error messages if pipeline fails.

---

## ✅ Success Indicators

Pipeline completes successfully when you see:

```
🎯 SUCCESS: Pipeline completed!
✓ PHASE 0 Complete: dataset_merged.csv
✓ PHASE 1 Complete
📊 Use outputs/latest/dataset_merged.csv for analysis!
```

Generated files in `outputs/latest/`:

- ✓ dataset_merged.csv (>500 KB)
- ✓ diagnostic_report.csv
- ✓ config.yaml
- ✓ 7+ PNG visualization files

---

## 🎓 Next Steps

1. **Explore outputs**: Check `outputs/latest/dataset_merged.csv`
2. **View visualizations**: Open PNG files to inspect data quality
3. **Review config**: Check `config.yaml` recommendations
4. **Use data**: Import `dataset_merged.csv` for your analysis
5. **Streamlit dashboards**: Use main `STREAMLIT_APP` for interactive exploration

---

## 📚 Related Documentation

- [Root README.md](../README.md) - Full project overview
- [CONFIG_GUIDE.md](./CONFIG_GUIDE.md) - Configuration details
- [Docker Setup](../DOCKER_SETUP.md) - Containerized execution
  ├── config_template.yaml # Configuration template (reference)
  ├── modules/
  │ ├── **init**.py
  │ ├── config_handler.py # Configuration management
  │ ├── data_integration.py # Phase 0: Data merging
  │ ├── diagnostics.py # Phase 1: Profiling & analysis
  │ ├── processing.py # Phase 2 preprocessing (retained for reference)
  │ └── logger_setup.py # Logging system
  │
  ├── outputs/
  │ ├── latest/ # Latest pipeline results
  │ ├── runs/ # Historical runs by timestamp
  │ └── scenarios/ # Different scenario runs
  │
  └── README.md # This file

# Input data location (must exist)

../nam/
├── Access to electricity (% of population).csv
├── Agriculture, forestry, and fishing, value added (% of GDP).csv
├── ... (other indicator CSVs)

```

---

## Phase-by-Phase Guide

### PHASE 0: Data Integration

**File:** `00_data_integration.ipynb`

**What it does:**

- Loads all CSV files from `../nam` folder
- Merges them based on Country and Year
- Standardizes structure: Countries (rows), Years (columns)
- Outputs: `dataset_merged.csv`

**How to run:**

```

1. Open 00_data_integration.ipynb
2. Run all cells (Shift+Enter)
3. Check output/dataset_merged.csv is created

```

**Expected output:**

- Shape: (countries × years, indicators)
- All indicators as columns
- Missing values marked as NaN

**Typical issues & solutions:**
| Issue | Solution |
|-------|----------|
| CSV not found | Verify CSV files in `../nam/` |
| Memory error | Check available RAM (reduce dataset size if needed) |
| Shape mismatch | Ensure all CSVs have Country names in rows |

---

### PHASE 1: Auto Profiling & Diagnostics

**File:** `01_auto_profiling.ipynb`

**What it does:**

- Analyzes data distributions for each column
- Detects missing values
- Identifies outliers (IQR method by default)
- Creates visualizations: boxplots, histograms
- Generates diagnostic report in CSV & JSON formats
- **Auto-generates draft configuration** (draft_config.yaml)

**How to run:**

```

1. Ensure PHASE 0 is complete (dataset_merged.csv exists)
2. Open 01_auto_profiling.ipynb
3. Run all cells
4. Review diagnostic reports and visualizations

````

**Generated files:**

- `diagnostic_report.csv` - Statistics table (Mean, Median, Std, Skewness, ...)
- `diagnostic_report.json` - Detailed results in JSON format
- `boxplots.png` - Boxplots for all columns (outlier detection)
- `histograms.png` - Histograms for all columns (distribution analysis)
- `draft_config.yaml` - Auto-generated configuration (**_ MUST REVIEW _**)

**Key metrics explained:**

- **Missing %:** Percentage of NaN values (>50% = problematic)
- **Skewness:** Distribution symmetry (-1 to 1 = moderate, >2 = highly skewed)
- **Outlier count:** Physical outliers detected by IQR method
- **Kurtosis:** Tail heaviness (>3 = heavy tails, unusual events)

---

## Output Files

### Phase 0 Outputs

| File                 | Type | Description                        |
| -------------------- | ---- | ---------------------------------- |
| `dataset_merged.csv` | CSV  | Merged dataset with all indicators |

### Phase 1 Outputs

| File                      | Type | Description                       |
| ------------------------- | ---- | --------------------------------- |
| `diagnostic_report.csv`   | CSV  | Summary statistics table          |
| `diagnostic_report.json`  | JSON | Detailed diagnostic results       |
| `boxplots.png`            | PNG  | Boxplot visualizations (~500KB)   |
| `histograms.png`          | PNG  | Histogram visualizations (~500KB) |
| `correlation_heatmap.png` | PNG  | Correlation matrix heatmap        |
| `pairwise_scatter.png`    | PNG  | Pairwise scatter plot matrix      |
| `config.yaml`             | YAML | Auto-generated recommendations    |

---

## Troubleshooting

### Issue: "Module not found" error

**Solution:**

```python
# Ensure modules folder is in Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'modules'))
````

### Issue: "FileNotFoundError: dataset_merged.csv"

**Solution:**

- Run PHASE 0 first (00_data_integration.ipynb)
- Verify CSV files exist in `../nam/` folder

### Issue: Memory error when running pipeline

**Solution:**

- Reduce dataset size (filter by years/countries)
- Process in chunks instead of all at once
- Check available RAM: `import psutil; psutil.virtual_memory()`

### Issue: "KeyError" in config loading

**Solution:**

- Verify config.yaml has correct YAML formatting
- Use online YAML validator: https://www.yamllint.com/
- Check indentation (use spaces, not tabs)

### Finding Logs from Previous Runs

**Solution:**

- All logs are automatically saved in `outputs/runs/run_TIMESTAMP_SCENARIO/logs/`
- Each run has 3 log files:
  - `pipeline.log` - Main pipeline execution
  - `data_integration.log` - PHASE 0 details
  - `diagnostics.log` - PHASE 1 analysis

**Example:**

```bash
# View logs from latest run
tail -f outputs/latest/logs/pipeline.log

# Find errors in all logs
grep "ERROR" outputs/runs/run_*/logs/*.log

# Check diagnostics steps
grep "Step" outputs/latest/logs/diagnostics.log
```

---

## Performance Notes

### Runtime Estimates

- **PHASE 0:** 10-60 seconds (depends on CSV sizes)
- **PHASE 1:** 30-120 seconds
- **Total:** 1-3 minutes for typical datasets

### Memory Usage

- CSV files in `../nam/`: ~50-200 MB
- Merged dataset: ~100-400 MB
- Diagnostic operations: In-place (minimal overhead)
- Visualizations: ~2-5 MB per file

### Data Considerations

- Maximum rows: 10,000+ (limited by available RAM)
- Numeric columns: 50+ supported
- Missing data: Reported in diagnostics
- Outliers: Detected using IQR, LOF, Mahalanobis, and Isolation Forest methods

---

## Tips & Best Practices

1. **Always review Phase 1 diagnostics** before further analysis
2. **Keep original data** - don't modify ../nam/ folder
3. **Check visualizations** - examine boxplots and histograms for outliers
4. **Use diagnostic reports** - review CSV and JSON outputs for insights
5. **Monitor logs** - check console output for warnings or errors

---

## References

- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **Scikit-learn Preprocessing:** https://scikit-learn.org/stable/modules/preprocessing.html
- **YAML Format:** https://yaml.org/spec/1.2/spec.html
- **Data Profiling Best Practices:** https://towardsdatascience.com/exploratory-data-analysis/

---

## Preprocessing Module Reference

The `processing.py` module contains utilities for data preprocessing (not executed in pipeline but available for programmatic use):

### Available Features (in processing.py):

- **Transform Operations**: Log transform, quantile transform, L1/L2 normalization
- **Outlier Detection**: IQR (univariate), LOF (multivariate), Mahalanobis, Isolation Forest
- **Outlier Handling**: Clip, drop, or impute outliers
- **Scaling Methods**: Standard, MinMax, Robust, IQR-based
- **Imputation Strategies**: Simple (mean/median), KNN, Iterative (Tree-based)

To use these for manual preprocessing:

```python
from modules.processing import DataProcessor

# Load data
df = pd.read_csv('dataset_merged.csv')

# Create processor
processor = DataProcessor(df)

# Apply transformations
processed_df, models = processor.run_processing_pipeline({
    'scaling': {'method': 'standard'},
    'imputation': {'method': 'knn', 'n_neighbors': 5},
})
```

---

**Questions or Issues?** Check the Troubleshooting section or review the notebook markdown cells for detailed explanations.
