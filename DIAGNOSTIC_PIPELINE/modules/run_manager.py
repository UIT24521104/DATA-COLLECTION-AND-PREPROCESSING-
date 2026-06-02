"""
Module: Run Manager
Quản lý cấu trúc outputs với hỗ trợ nhiều kịch bản (scenarios)

Kiến trúc outputs/:
├── scenarios/                      # Lưu config cho các kịch bản khác nhau
│   ├── baseline_config.yaml
│   └── conservative_config.yaml
├── runs/                           # Lưu kết quả từng lần chạy
│   ├── run_20260411_143000_baseline/
│   └── run_20260411_145200_conservative/
└── latest/                         # Kết quả run mới nhất (copy files)
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class RunManager:
    """Quản lý cấu trúc outputs cho nhiều scenarios"""
    
    def __init__(self, outputs_dir: str = './outputs'):
        """
        Khởi tạo RunManager
        
        Args:
            outputs_dir: Đường dẫn folder outputs
        """
        self.outputs_dir = Path(outputs_dir)
        self.scenarios_dir = self.outputs_dir / 'scenarios'
        self.runs_dir = self.outputs_dir / 'runs'
        self.latest_dir = self.outputs_dir / 'latest'
        
        # Tạo các folder cơ bản
        self._setup_directories()
    
    def _setup_directories(self):
        """Tạo cấu trúc thư mục cơ bản"""
        self.outputs_dir.mkdir(exist_ok=True)
        self.scenarios_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Setup output directories: {self.outputs_dir}")
    
    def create_run_directory(self, scenario_name: str = 'default') -> Path:
        """
        Tạo folder cho một lần chạy mới với timestamp
        
        Args:
            scenario_name: Tên kịch bản (e.g., 'baseline', 'conservative')
        
        Returns:
            Path đến folder run mới
        """
        # Tạo timestamp: 20260411_143000
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Folder name: run_20260411_143000_baseline
        run_folder_name = f'run_{timestamp}_{scenario_name}'
        run_folder = self.runs_dir / run_folder_name
        
        # Tạo folder
        run_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Created run directory: {run_folder}")
        
        return run_folder
    
    def save_config_to_run(self, run_dir: Path, config: Dict[str, Any], config_name: str = 'config.yaml'):
        """
        Lưu config YAML vào folder run
        
        Args:
            run_dir: Folder của lần chạy
            config: Dict config
            config_name: Tên file config
        """
        import yaml
        
        config_path = run_dir / config_name
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✓ Saved config: {config_path}")
        return config_path
    
    def copy_scenario_config(self, scenario_config_name: str, run_dir: Path) -> Path:
        """
        Copy config từ folder scenarios vào run folder
        
        Args:
            scenario_config_name: Tên file config trong scenarios/ (e.g., 'baseline_config.yaml')
            run_dir: Folder của lần chạy
        
        Returns:
            Path đến config file đã copy
        """
        source = self.scenarios_dir / scenario_config_name
        dest = run_dir / 'config.yaml'
        
        if source.exists():
            shutil.copy(source, dest)
            logger.info(f"✓ Copied config: {source} → {dest}")
            return dest
        else:
            logger.warning(f"⚠ Scenario config not found: {source}")
            return None
    
    def update_latest(self, run_dir: Path):
        """
        Update folder 'latest' với kết quả từ run mới nhất
        Copy tất cả files từ run_dir vào latest/
        
        Args:
            run_dir: Folder của lần chạy mới nhất
        """
        import shutil
        import os
        
        # Strategy: Copy files selectively instead of replacing entire folder
        # This avoids file locking issues on Windows
        
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all items from run_dir to latest
        for item in run_dir.iterdir():
            dst_path = self.latest_dir / item.name
            
            if item.is_dir():
                # Remove destination if exists (with retry for locked files)
                if dst_path.exists():
                    try:
                        shutil.rmtree(dst_path, ignore_errors=True)
                    except:
                        pass  # Continue even if we can't remove
                
                # Copy directory
                try:
                    shutil.copytree(item, dst_path, dirs_exist_ok=True)
                except Exception as e:
                    logger.warning(f"Could not copy directory {item.name}: {e}")
            else:
                # Copy file (overwrite if exists)
                try:
                    shutil.copy2(item, dst_path)
                except Exception as e:
                    logger.warning(f"Could not copy file {item.name}: {e}")
        
        logger.info(f"✓ Updated latest directory: {self.latest_dir}")
    
    def get_run_list(self) -> list:
        """
        Liệt kê tất cả runs (sắp xếp theo thời gian mới nhất)
        
        Returns:
            List of (run_name, run_path) tuples
        """
        if not self.runs_dir.exists():
            return []
        
        runs = sorted([d for d in self.runs_dir.iterdir() if d.is_dir()], reverse=True)
        return [(r.name, r) for r in runs]
    
    def get_run_info(self, run_dir: Path) -> Dict[str, Any]:
        """
        Lấy thông tin về một run: tên scenario, timestamp, file outputs
        
        Args:
            run_dir: Folder của lần chạy
        
        Returns:
            Dict với thông tin run
        """
        run_name = run_dir.name
        
        # Parse run_name: run_20260411_143000_baseline
        parts = run_name.split('_')
        if len(parts) >= 3:
            timestamp_str = f"{parts[1]}_{parts[2]}"
            scenario = '_'.join(parts[3:])
        else:
            timestamp_str = 'unknown'
            scenario = 'unknown'
        
        # Liệt kê files
        files = {
            'config': (run_dir / 'config.yaml').exists(),
            'dataset_merged': (run_dir / 'dataset_merged.csv').exists(),
            'dataset_final': (run_dir / 'dataset_final.csv').exists(),
            'diagnostic_report_csv': (run_dir / 'diagnostic_report.csv').exists(),
            'diagnostic_report_json': (run_dir / 'diagnostic_report.json').exists(),
            'boxplots': (run_dir / 'boxplots.png').exists(),
            'histograms': (run_dir / 'histograms.png').exists(),
            'scaler_model': (run_dir / 'scaler_model.pkl').exists(),
            'processing_metadata': (run_dir / 'processing_metadata.json').exists(),
            'run_summary': (run_dir / 'run_summary.json').exists(),
        }
        
        # Tính tổng kích thước
        total_size = sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file())
        
        return {
            'run_name': run_name,
            'timestamp': timestamp_str,
            'scenario': scenario,
            'files': files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'path': str(run_dir),
        }
    
    def save_run_summary(self, run_dir: Path, summary: Dict[str, Any]):
        """
        Lưu tóm tắt lần chạy vào JSON file
        
        Args:
            run_dir: Folder của lần chạy
            summary: Dict với thông tin tóm tắt (thời gian, shapes, metrics, ...)
        """
        summary_path = run_dir / 'run_summary.json'
        
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'run_info': self.get_run_info(run_dir),
            'summary': summary,
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved run summary: {summary_path}")
        return summary_path
    
    def list_scenarios(self) -> list:
        """
        Liệt kê tất cả scenario configs
        
        Returns:
            List of scenario config file names
        """
        if not self.scenarios_dir.exists():
            return []
        
        return [f.name for f in self.scenarios_dir.glob('*.yaml')]
    
    def compare_runs(self, run_dirs: list) -> Dict[str, Any]:
        """
        So sánh nhiều runs
        
        Args:
            run_dirs: List of Path objects
        
        Returns:
            Dict so sánh các runs
        """
        comparison = {}
        
        for run_dir in run_dirs:
            info = self.get_run_info(run_dir)
            comparison[info['scenario']] = {
                'timestamp': info['timestamp'],
                'size_mb': info['total_size_mb'],
                'files_present': sum(1 for v in info['files'].values() if v),
            }
        
        return comparison


def print_run_list(run_manager: RunManager):
    """Hiển thị danh sách tất cả runs"""
    runs = run_manager.get_run_list()
    
    if not runs:
        print("No runs found.")
        return
    
    print("\n" + "="*80)
    print("RUN LIST (newest first)")
    print("="*80)
    
    for i, (run_name, run_path) in enumerate(runs, 1):
        info = run_manager.get_run_info(run_path)
        print(f"\n{i}. {info['scenario'].upper()}")
        print(f"   Timestamp: {info['timestamp']}")
        print(f"   Size: {info['total_size_mb']} MB")
        print(f"   Files: {sum(1 for v in info['files'].values() if v)}/{len(info['files'])}")
        print(f"   Path: {run_path}")


if __name__ == '__main__':
    # Test
    rm = RunManager()
    print_run_list(rm)
