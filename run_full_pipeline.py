#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
🚀 WORLDBANK DATA ANALYSIS - MASTER PIPELINE RUNNER
=============================================================================
Master script để chạy toàn bộ pipeline từ Phase 0 đến Phase 2

Usage:
    python run_full_pipeline.py                    # Chạy mặc định (default scenario)
    python run_full_pipeline.py baseline           # Chạy scenario 'baseline'
    python run_full_pipeline.py conservative       # Chạy scenario 'conservative'

Output:
    - Toàn bộ kết quả được lưu vào: outputs/runs/run_TIMESTAMP_SCENARIO/
    - Latest results được copy vào: outputs/latest/
    - Logs được lưu vào: outputs/runs/run_TIMESTAMP_SCENARIO/logs/

=============================================================================
"""

import sys
import os
from pathlib import Path
import logging

# Fix Unicode encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup paths
current_dir = Path(__file__).parent
pipeline_dir = current_dir / 'pipeline_execution'
sys.path.insert(0, str(pipeline_dir / 'modules'))
sys.path.insert(0, str(pipeline_dir))
os.chdir(pipeline_dir)

# Import và chạy
from run_pipeline import main

if __name__ == '__main__':
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else 'default'
    
    print(f"\n{'='*80}")
    print(f"🚀 Chạy Pipeline với Scenario: {scenario_name.upper()}")
    print(f"{'='*80}\n")
    
    success = main(scenario_name=scenario_name)
    exit_code = 0 if success else 1
    print(f"\nExit code: {exit_code} (success={success})")
    sys.exit(exit_code)
