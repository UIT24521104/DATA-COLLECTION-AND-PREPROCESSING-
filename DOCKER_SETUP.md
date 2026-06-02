# 🐳 Docker Setup Guide - WorldBank Analysis Project

## Overview

This Dockerfile supports **3 independent commands**, each running a different component of the project:

1. **DIAGNOSTIC_PIPELINE**: Data integration & auto-profiling
2. **STREAMLIT_APP**: Interactive dashboard
3. **TRAJECTORY_APP**: Trajectory similarity analysis

---

## Prerequisites

- Docker Desktop installed and running
- Project files: `requirements.txt`, `DIAGNOSTIC_PIPELINE/`, `STREAMLIT_APP/`, `TRAJECTORY_APP/`

---

## Build Image

```bash
cd e:\WorldBank-Data-Analysis-Project
docker build -t worldbank:latest .
```

---

## Command Reference

### 1️⃣ DIAGNOSTIC PIPELINE

**Purpose**: Merge CSV files & generate diagnostics

```bash
# Default scenario
docker run worldbank:latest diagnostic default

# Custom scenario
docker run worldbank:latest diagnostic test

# Persist outputs to local machine (Windows)
docker run -v %cd%\outputs:/app/DIAGNOSTIC_PIPELINE/outputs ^
  worldbank:latest diagnostic default

# Persist outputs to local machine (Linux/Mac)
docker run -v $(pwd)/outputs:/app/DIAGNOSTIC_PIPELINE/outputs \
  worldbank:latest diagnostic default
```

**Output**: `DIAGNOSTIC_PIPELINE/outputs/runs/<timestamp>/`

---

### 2️⃣ STREAMLIT APP (Main Dashboard)

**Purpose**: Interactive data exploration & diagnostics

```bash
# Default port 8501
docker run -p 8501:8501 worldbank:latest streamlit

# Custom port (e.g., 9000)
docker run -p 9000:8501 worldbank:latest streamlit 8501
```

**Access**: Open browser → `http://localhost:8501`

**Features**:

- Data explorer & filtering
- EDA visualizations
- Diagnostics reports
- Outlier detection

---

### 3️⃣ TRAJECTORY APP (Similarity Analysis)

**Purpose**: Compare Vietnam with similar countries

```bash
# Default port 8502
docker run -p 8502:8502 worldbank:latest trajectory

# Custom port (e.g., 9001)
docker run -p 9001:8502 worldbank:latest trajectory 8502
```

**Access**: Open browser → `http://localhost:8502`

**Features**:

- Country similarity ranking
- Trajectory visualization
- Economic indicators comparison

---

## Common Use Cases

### ✅ Run Pipeline Only (Batch Processing)

```bash
docker run -v %cd%\outputs:/app/DIAGNOSTIC_PIPELINE/outputs ^
  worldbank:latest diagnostic default
```

### ✅ Run Both Dashboards (Separate Terminals)

**Terminal 1** - STREAMLIT_APP:

```bash
docker run -p 8501:8501 worldbank:latest streamlit
```

**Terminal 2** - TRAJECTORY_APP:

```bash
docker run -p 8502:8502 worldbank:latest trajectory
```

### ✅ Run with Custom Ports (Multiple Instances)

```bash
# Instance 1 - Port 9000
docker run -p 9000:8501 worldbank:latest streamlit

# Instance 2 - Port 9001
docker run -p 9001:8502 worldbank:latest trajectory
```

### ✅ Debug Mode (Shell Access)

```bash
docker run -it worldbank:latest /bin/bash

# Inside container:
# cd DIAGNOSTIC_PIPELINE && python run_pipeline.py full --scenario default
# cd STREAMLIT_APP && streamlit run app.py --server.port 8501
```

### ✅ Run with Mounted Data

```bash
# Mount local data folder
docker run -v %cd%\DATA_AFTER_PREPROCESSING:/app/DATA_AFTER_PREPROCESSING ^
  -p 8501:8501 ^
  worldbank:latest streamlit
```

---

## Troubleshooting

### 📍 Port Already in Use

```bash
# Change port
docker run -p 9000:8501 worldbank:latest streamlit

# Or kill existing container
docker ps
docker kill <container_id>
```

### 📍 Pipeline Outputs Not Saved

Use volume mount with `-v`:

```bash
docker run -v %cd%\outputs:/app/DIAGNOSTIC_PIPELINE/outputs ^
  worldbank:latest diagnostic default
```

### 📍 Streamlit App Won't Connect

Check logs:

```bash
docker logs <container_id>
```

Then access with correct port:

- STREAMLIT_APP: `http://localhost:8501`
- TRAJECTORY_APP: `http://localhost:8502`

### 📍 Data Not Found Error

Ensure volume mounts are correct:

```bash
# Windows
docker run -v C:\full\path\to\data:/app/DATA_AFTER_PREPROCESSING ^
  -p 8501:8501 worldbank:latest streamlit

# Linux/Mac
docker run -v /full/path/to/data:/app/DATA_AFTER_PREPROCESSING \
  -p 8501:8501 worldbank:latest streamlit
```

---

## Show Help in Docker

```bash
docker run worldbank:latest help
```

Shows all available commands with detailed examples.

---

## Container Management

```bash
# List running containers
docker ps

# View logs
docker logs <container_id>

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>

# Remove image
docker rmi worldbank:latest
```

---

## Performance Tips

- **Pipeline**: Volume mount outputs to avoid storing in container
- **Streamlit**: Run on separate containers/ports for multiple instances
- **Memory**: Add `docker run --memory="2g"` if needed

---

## Supported Aliases

| Command    | Aliases                          |
| ---------- | -------------------------------- |
| diagnostic | `pipeline`, `diagnostic`         |
| streamlit  | `streamlit`, `dashboard`, `main` |
| trajectory | `trajectory`                     |
| help       | `help`, `--help`, `-h`           |
