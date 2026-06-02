# 📚 HƯỚNG DẪN TẠO CONDA ENVIRONMENT & CHẠY PIPELINE

## 1️⃣ Chuẩn Bị (Prerequisites)

### Khi chưa có Python/Conda

- **Windows**: Tải Miniconda từ https://docs.conda.io/projects/miniconda/en/latest/
- **macOS/Linux**: Tương tự
- Sau khi câi, khởi động lại terminal

### Kiểm tra cài đặt

```bash
# Windows PowerShell / CMD
conda --version
python --version
```

---

## 2️⃣ TẠO CONDA ENVIRONMENT

### Bước 1: Mở Terminal

- **Windows**: Mở PowerShell hoặc CMD
- **macOS/Linux**: Mở Terminal

### Bước 2: Điều hướng tới thư mục dự án

```bash
cd e:\WorldBank-Data-Analysis-Project

# Hoặc nếu dùng macOS/Linux
cd ~/path/to/WorldBank-Data-Analysis-Project
```

### Bước 3: Tạo Conda Environment

#### **Cách 1️⃣: Tạo từ environment.yml** (Nên dùng - đơn giản nhất)

```bash
conda env create -f environment.yml
```

#### **Cách 2️⃣: Tạo thủ công từ requirements.txt**

```bash
# Tạo environment mới với Python 3.11
conda create -n worldbank-analysis python=3.11 -y

# Activate environment
conda activate worldbank-analysis

# Install packages
pip install -r requirements.txt
```

#### **Cách 3️⃣: Tạo bằng venv** (Nếu không có conda)

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Bước 4: Kiểm tra Environment

```bash
# Activate environment (nếu chưa)
conda activate worldbank-analysis

# Kiểm tra packages
pip list

# Test imports
python -c "import pandas; import numpy; import sklearn; print('✓ All packages OK')"
```

---

## 3️⃣ CHẠY PIPELINE

### Kích hoạt Environment (mỗi lần làm việc)

```bash
conda activate worldbank-analysis

# Hoặc nếu dùng venv:
.\.venv\Scripts\activate          # Windows PowerShell
.\.venv\Scripts\activate.bat       # Windows CMD
source .venv/bin/activate          # macOS/Linux
```

### Chạy Pipeline từ Project Root

```bash
# Chạy với mặc định
python run_full_pipeline.py

# Hoặc chạy với scenario cụ thể
python run_full_pipeline.py baseline
python run_full_pipeline.py conservative
```

### Hoặc chạy từ folder pipeline_execution

```bash
cd pipeline_execution
python run_pipeline.py default
cd ..
```

---

## 4️⃣ KIỂM TRA KẾT QUẢ

Tất cả output được lưu vào: **`outputs/`**

```
outputs/
├── runs/                              # Tất cả lần chạy
│   ├── run_20260413_100000_default/   # Lần chạy 1
│   ├── run_20260413_101500_baseline/  # Lần chạy 2
│   └── ...
├── latest/                            # Copy mới nhất
│   ├── dataset_final.csv
│   ├── diagnostic_report.csv
│   └── ...
└── scenarios/                         # Config các scenario
```

### File quan trọng nhất

- 📊 **`outputs/latest/dataset_final.csv`** - Dữ liệu đã xử lý (dùng cho ML models)
- 📋 **`outputs/latest/diagnostic_report.csv`** - Báo cáo chẩn đoán dữ liệu
- 📈 **`outputs/latest/boxplots.png`** - Biểu đồ phân phối
- 📉 **`outputs/latest/histograms.png`** - Histogram

---

## 5️⃣ TROUBLESHOOTING

### ❌ Python không tìm thấy

```bash
# Tìm đường dẫn Python
which python          # macOS/Linux
where python          # Windows

# Hoặc chỉ định đường dẫn đầy đủ
C:\Users\YourName\Miniconda3\envs\worldbank-analysis\python.exe run_full_pipeline.py
```

### ❌ Package import error

```bash
# Reinstall packages
pip install --upgrade -r requirements.txt

# Hoặc dùng conda
conda install -c conda-forge scikit-learn pandas numpy matplotlib
```

### ❌ Permission denied (macOS/Linux)

```bash
# Cấp quyền executable
chmod +x run_full_pipeline.py
python run_full_pipeline.py
```

### ❌ Error: "No module named 'xxxx'"

```bash
# Kiểm tra environment active
conda info | grep active

# Nếu không đúng, activate lại
conda activate worldbank-analysis

# Reinstall
pip install -r requirements.txt
```

---

## 6️⃣ XÓA ENVIRONMENT (Nếu cần)

```bash
# Xóa environment
conda env remove -n worldbank-analysis

# Xóa virtual folder (nếu dùng venv)
rm -r .venv                   # macOS/Linux
rmdir /s .venv                # Windows
```

---

## 📝 Notes

- **Lần đầu chạy**: Sẽ tải data từ `nam/` folder, merge và xử lý (5-10 phút)
- **Logged data**: Xem chi tiết ở `outputs/runs/run_XXX/logs/`
- **Hỗ trợ Python**: 3.9, 3.10, 3.11, 3.12

---

**Questions? Check:**

- 📖 [README.md](pipeline_execution/README.md) - Pipeline documentation
- 📊 [outputs/latest/](outputs/latest/) - Latest results
