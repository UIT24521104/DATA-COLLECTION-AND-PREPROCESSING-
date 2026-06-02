"""
Master Pipeline Execution Script
=================================
Three explicit phases:
PHASE 0: Merge data
PHASE 1: Diagnose merged data and generate draft config
PHASE 2: Preprocess merged data using reviewed config

Usage:
    python run_pipeline.py full
    python run_pipeline.py phase0
    python run_pipeline.py phase1
    python run_pipeline.py phase2
    python run_pipeline.py <scenario_name>   # Backward-compatible full run
"""

import sys
import os
from pathlib import Path
import logging
import json
import argparse
from typing import Optional

# Setup paths
pipeline_dir = Path(__file__).parent
sys.path.insert(0, str(pipeline_dir / 'modules'))
os.chdir(pipeline_dir)

# Import modules
from data_integration import DataIntegration
from diagnostics import DataDiagnostics
from config_handler import ConfigHandler
from run_manager import RunManager
from logger_setup import LoggerSetup

import pandas as pd
import yaml
import numpy as np


def run_phase_0(run_dir: Path, logger: logging.Logger, logger_setup: LoggerSetup):
    """PHASE 0: Data Integration"""
    logger.info("\n" + "="*70)
    logger.info("PHASE 0: DATA INTEGRATION")
    logger.info("="*70)
    
    # Use absolute path from project root
    input_folder = str(Path(__file__).parent.parent / 'RAW_DATASET')
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Setup module logger for data_integration
    module_logger = logger_setup.get_module_logger(
        'data_integration',
        log_file='data_integration.log'
    )
    
    # Run integration
    integration = DataIntegration(input_folder=input_folder, logger=module_logger)
    merged_df = integration.run_integration()
    
    # Filter out non-country territories at Phase 0
    non_countries = [
        'HKG', 'MAC', 'PRI', 'GRL', 'BMU', 'VIR', 'VGB', 'CYM', 'TCA', 'ABW', 'CUW', 'SXM', 'MAF',
        'FRO', 'GIB', 'IMN', 'CHI', 'GUM', 'NCL', 'PYF', 'MNP', 'ASM', 'PSE', 'XKX',
        'WLD', 'HIC', 'UMC', 'MIC', 'LMC', 'LIC', 'LDC', 'HPC', 'FCS', 'IDA', 'IBD', 'IBT', 'IDX', 'IDB',
        'EAR', 'PRE', 'PST', 'SST', 'OSS', 'EUU', 'EMU', 'OED', 'ARB',
        'AFE', 'AFW', 'SSF', 'SSA', 'LCN', 'LAC', 'TLA', 'NAC', 'CSS',
        'ECS', 'ECA', 'TEC', 'CEB', 'EAS', 'EAP', 'TEA', 'SAS', 'TSA', 'PSS', 'MEA', 'MNA', 'TMN'
    ]
    
    initial_shape = merged_df.shape
    merged_df = merged_df[~merged_df['Country'].isin(non_countries)].copy()
    removed_count = initial_shape[0] - merged_df.shape[0]
    removed_pct = (removed_count / initial_shape[0] * 100) if initial_shape[0] > 0 else 0
    
    logger.info(f"✓ Filtered non-country territories:")
    logger.info(f"  Before: {initial_shape[0]} rows")
    logger.info(f"  After: {merged_df.shape[0]} rows")
    logger.info(f"  Removed: {removed_count} rows ({removed_pct:.1f}%)")
    
    # Save to run directory
    output_path = run_dir / 'dataset_merged.csv'
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"✓ PHASE 0 Complete: {output_path}")
    logger.info(f"  Shape: {merged_df.shape}")
    logger.info(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    return merged_df


def run_phase_1(df: pd.DataFrame, run_dir: Path, logger: logging.Logger, logger_setup: LoggerSetup):
    """PHASE 1: Auto Profiling & Diagnostics"""
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: AUTO PROFILING & DIAGNOSTICS")
    logger.info("="*70)
    
    # Setup module logger for diagnostics
    module_logger = logger_setup.get_module_logger(
        'diagnostics',
        log_file='diagnostics.log'
    )
    
    # Run diagnostics
    diagnostics = DataDiagnostics(df, logger=module_logger)
    diagnostic_report = diagnostics.run_diagnostics(
        outlier_method='iqr',
        iqr_multiplier=1.5
    )
    
    # Save reports to run directory (CSV only)
    diagnostics.save_report_csv(str(run_dir / 'diagnostic_report.csv'))
    
    # Create visualizations
    logger.info("Creating visualizations...")
    diagnostics.create_visualizations(
        output_folder=str(run_dir),
        figsize=(15, 12)
    )
    
    # Create draft config
    logger.info("Creating draft configuration...")
    config_handler = ConfigHandler()
    draft_config = config_handler.create_default_config()
    
    # Update recommendations based on diagnostics
    outlier_info = diagnostic_report.get('outlier_detection', {})
    total_outliers = outlier_info.get('total_outlier_records', 0)
    total_records = len(df)
    outlier_percentage = (total_outliers / total_records * 100) if total_records > 0 else 0
    
    logger.info(f"  Outlier percentage: {outlier_percentage:.2f}%")
    
    if outlier_percentage > 5:
        logger.info("  → Recommending higher IQR multiplier (1.8)")
        draft_config['phase1']['outlier_detection']['iqr_multiplier'] = 1.8
    
    # Check for skewed distributions
    dist_report = diagnostic_report.get('distribution_analysis', {})
    skewed_cols = [col for col, stats in dist_report.items() if abs(stats['skewness']) > 2]
    if skewed_cols:
        logger.info(f"  → Found {len(skewed_cols)} highly skewed columns")
        logger.info(f"  → Recommending LOG TRANSFORM")
        draft_config['phase1']['log_transform']['enabled'] = True
    
    logger.info(f"✓ PHASE 1 Complete")
    logger.info(f"  Reports: diagnostic_report.csv")
    logger.info(f"  Visualizations: boxplots.png, histograms.png, correlation_heatmap.png, pairwise_scatter.png")
    logger.info(f"  Draft config: config.yaml (for phase 2 review)")
    
    return draft_config


def run_phase_2(df: pd.DataFrame, config: dict, run_dir: Path, logger: logging.Logger, logger_setup: LoggerSetup):
    """PHASE 2: Preprocessing Pipeline (NOT IMPLEMENTED - FOR REFERENCE ONLY)"""
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: PREPROCESSING (SKIPPED - NOT IMPLEMENTED)")
    logger.info("="*70)
    logger.warning("⊘ Phase 2 preprocessing is not available in this version.")
    logger.warning("  Use outputs/latest/dataset_merged.csv for manual preprocessing.")
    return df


def main(scenario_name: str = 'default'):
    """
    Main execution - Chạy toàn bộ pipeline
    
    Args:
        scenario_name: Tên kịch bản (được truyền từ command line)
    """
    try:
        # Initialize RunManager
        run_manager = RunManager('./outputs')
        # Tạo định danh: Sử dụng RunManager để tạo một thư mục riêng biệt trong outputs/runs/ dựa trên thời gian thực (ví dụ: run_20260411_...). Điều này giúp bạn không bao giờ bị ghi đè kết quả cũ.
        # Create run directory
        run_dir = run_manager.create_run_directory(scenario_name=scenario_name)
        
        # Thiết lập nhật ký (Logging): Tạo các file log riêng cho từng công đoạn (data_integration.log, diagnostics.log,...) để dễ dàng kiểm tra lỗi sau này.
        # Setup logging for this run
        logger_setup = LoggerSetup(run_dir)
        logger = logger_setup.configure_root_logger(level=logging.INFO)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"WORLDBANK DATA PROCESSING PIPELINE")
        logger.info(f"Scenario: {scenario_name}")
        logger.info(f"{'='*70}\n")
        
        logger.info(f"📁 Run Directory: {run_dir}")
        logger.info(f"📝 Logs Directory: {logger_setup.logs_dir}")
        
        # PHASE 0
        merged_df = run_phase_0(run_dir, logger, logger_setup)
        
        # PHASE 1
        draft_config = run_phase_1(merged_df, run_dir, logger, logger_setup)
        
        # Save config to run directory
        run_manager.save_config_to_run(run_dir, draft_config, 'config.yaml')
        
        # NOTE: Phase 2 (preprocessing) has been removed from automated pipeline
        # The DataProcessor and preprocessing modules are retained for reference
        # and can be used programmatically if needed.
        
        # Save run summary
        summary = {
            'scenario': scenario_name,
            'merged_shape': list(merged_df.shape),
            'phases_completed': ['PHASE 0', 'PHASE 1'],
        }
        run_manager.save_run_summary(run_dir, summary)
        
        # Update latest directory
        run_manager.update_latest(run_dir)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("🎉 PIPELINE EXECUTION COMPLETE (Phase 0 & 1)")
        logger.info("="*70)
        logger.info(f"\n📂 Run Directory: {run_dir}")
        logger.info(f"\n📋 Generated Files:")
        
        for file in sorted(run_dir.glob('*')):
            if file.is_file():
                size = file.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size / (1024*1024):.2f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.2f} KB"
                else:
                    size_str = f"{size} B"
                logger.info(f"  ✓ {file.name:<35} {size_str:>12}")
        
        # Logs info
        logger.info(f"\n📝 Logs Files:")
        for log_file in sorted(logger_setup.logs_dir.glob('*.log')):
            size = log_file.stat().st_size
            if size > 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size} B"
            logger.info(f"  ✓ {log_file.name:<35} {size_str:>12}")
        
        logger.info("\n" + "="*70)
        logger.info(f"📁 Latest run: {run_manager.latest_dir}")
        logger.info(f"📊 Use outputs/latest/dataset_merged.csv for analysis!")
        logger.info("="*70)
        
        # Print run list
        logger.info("\n📚 ALL RUNS:\n")
        try:
            runs = run_manager.get_run_list()
            for i, (run_name, run_path) in enumerate(runs[:5], 1):  # Show top 5
                try:
                    info = run_manager.get_run_info(run_path)
                    logger.info(f"  {i}. {info['scenario'].upper():<20} | {info['timestamp']} | {info['total_size_mb']} MB")
                except Exception as e:
                    logger.warning(f"Could not get run info for {run_name}: {e}")
            
            if len(runs) > 5:
                logger.info(f"  ... and {len(runs)-5} more runs")
        except Exception as e:
            logger.warning(f"Could not retrieve run list: {e}")
        
        logger.info("RETURNING TRUE FROM MAIN")
        print("🎯 SUCCESS: Pipeline completed!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        print(f"❌ FAILED: {e}")
        return False


def resolve_phase1_input_file(input_file: Optional[str]) -> Path:
    """Resolve merged dataset path for standalone phase 1 diagnostics."""
    if input_file:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {input_path}")
        return input_path

    candidates = [
        Path('./outputs/latest/dataset_merged.csv'),
        Path('./outputs/dataset_merged.csv'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find dataset_merged.csv. Run phase0 first or pass --input <path>."
    )


def resolve_phase1_config_file(config_file: Optional[str]) -> Path:
    """Resolve YAML config path for standalone phase 2 preprocessing."""
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    candidates = [
        Path('./outputs/latest/final_config.yaml'),
        Path('./outputs/latest/config.yaml'),
        Path('./outputs/final_config.yaml'),
        Path('./outputs/draft_config.yaml'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find a config YAML file. Run phase0 first or pass --config <path>."
    )


def run_phase_0_standalone(scenario_name: str = 'default'):
    """Run phase 0 only: data integration."""
    try:
        run_manager = RunManager('./outputs')
        run_dir = run_manager.create_run_directory(scenario_name=f"{scenario_name}_phase0")

        logger_setup = LoggerSetup(run_dir)
        logger = logger_setup.configure_root_logger(level=logging.INFO)

        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 0 ONLY | Scenario: {scenario_name}")
        logger.info(f"{'='*70}\n")

        merged_df = run_phase_0(run_dir, logger, logger_setup)

        summary = {
            'scenario': scenario_name,
            'merged_shape': list(merged_df.shape),
            'phases_completed': ['PHASE 0 - MERGE'],
            'next_step': 'Run phase1 diagnostics on dataset_merged.csv',
        }
        run_manager.save_run_summary(run_dir, summary)
        run_manager.update_latest(run_dir)

        logger.info("\n✓ Standalone phase0 completed successfully")
        logger.info(f"  Run directory: {run_dir}")
        logger.info("  Generated file: dataset_merged.csv")

        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"\n❌ ERROR (phase0): {e}", exc_info=True)
        return False


def run_phase_1_standalone(
    scenario_name: str = 'default',
    input_file: Optional[str] = None,
):
    """Run phase 1 only: diagnostics on merged data + draft config generation."""
    try:
        run_manager = RunManager('./outputs')
        run_dir = run_manager.create_run_directory(scenario_name=f"{scenario_name}_phase1")

        logger_setup = LoggerSetup(run_dir)
        logger = logger_setup.configure_root_logger(level=logging.INFO)

        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 1 ONLY | Scenario: {scenario_name}")
        logger.info(f"{'='*70}\n")

        input_path = resolve_phase1_input_file(input_file)
        logger.info(f"Using merged dataset: {input_path}")

        merged_df = pd.read_csv(input_path)

        draft_config = run_phase_1(merged_df, run_dir, logger, logger_setup)

        run_manager.save_config_to_run(run_dir, draft_config, 'draft_config.yaml')
        run_manager.save_config_to_run(run_dir, draft_config, 'config.yaml')

        summary = {
            'scenario': scenario_name,
            'input_dataset': str(input_path),
            'merged_shape': list(merged_df.shape),
            'phases_completed': ['PHASE 1 - DIAGNOSTICS'],
            'next_step': 'Review config.yaml, then run phase2 preprocessing',
        }
        run_manager.save_run_summary(run_dir, summary)
        run_manager.update_latest(run_dir)

        logger.info("\n✓ Standalone phase1 completed successfully")
        logger.info(f"  Run directory: {run_dir}")
        logger.info("  Generated files: diagnostic_report.csv, boxplots.png, histograms.png, correlation_heatmap.png, pairwise_scatter.png")
        logger.info("  Generated config: draft_config.yaml / config.yaml")

        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"\n❌ ERROR (phase1): {e}", exc_info=True)
        return False


def parse_cli_args() -> argparse.Namespace:
    """Parse CLI args for separate phase execution."""
    parser = argparse.ArgumentParser(
        description='WorldBank pipeline runner with separate phase modes.'
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='full',
        choices=['full', 'phase0', 'phase1'],
        help='Execution mode: full (phase0+1), phase0 (merge only), phase1 (diagnostics only).',
    )
    parser.add_argument(
        '--scenario',
        default='default',
        help='Scenario name used for run folder naming.',
    )
    parser.add_argument(
        '--input',
        default=None,
        help='Path to dataset_merged.csv. Optional if default paths exist.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Backward-compatible mode: python run_pipeline.py <scenario_name>
    # when <scenario_name> is not one of supported mode names.
    if len(sys.argv) == 2 and sys.argv[1] not in {'full', 'phase0', 'phase1', '-h', '--help'}:
        success = main(scenario_name=sys.argv[1])
        sys.exit(0 if success else 1)

    args = parse_cli_args()

    if args.mode == 'full':
        success = main(scenario_name=args.scenario)
    elif args.mode == 'phase0':
        success = run_phase_0_standalone(scenario_name=args.scenario)
    elif args.mode == 'phase1':
        success = run_phase_1_standalone(
            scenario_name=args.scenario,
            input_file=args.input,
        )

    sys.exit(0 if success else 1)
