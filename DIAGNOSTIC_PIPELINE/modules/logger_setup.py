"""
Module: Logger Setup
Quản lý cấu hình logging cho toàn bộ pipeline
- Lưu logs vào file cho mỗi module
- Hiển thị logs trên console
- Cấu trúc logs: outputs/runs/run_TIMESTAMP_SCENARIO/logs/
"""
# Logger -> Log Record -> Handler -> Formatter
# logging.getLogger(name): Khởi tạo hoặc lấy lại 1 logger
# logger.setLevel(level): Thiết lập ngưỡng cho log
# logger.addHandler(hdlr): Thêm hanlder cho logger
# logger.removeHandler(hdlr): Loại bỏ handler thêm trước đó
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerSetup:
    """Quản lý cấu hình logging tập trung"""
    
    def __init__(self, run_dir: Path):
        """
        Khởi tạo LoggerSetup
        
        Args:
            run_dir: Đường dẫn folder run (e.g., outputs/runs/run_20260412_143000_baseline)
        """
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / 'logs'
        
        # Tạo folder logs
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo path cho các log files
        self.main_log_path = self.logs_dir / 'pipeline.log'
        self.data_integration_log_path = self.logs_dir / 'data_integration.log'
        self.diagnostics_log_path = self.logs_dir / 'diagnostics.log'
        self.processing_log_path = self.logs_dir / 'processing.log'
    
    def configure_root_logger(self, level: int = logging.INFO) -> logging.Logger:
        """
        Cấu hình root logger (dùng cho main pipeline)
        
        Args:
            level: Log level (INFO, DEBUG, etc.)
        
        Returns:
            Root logger instance
        """
        root_logger = logging.getLogger() 
        # Khi không truyền tên vào hàm này thì sẽ tạo ra một logger chung cho cả hệ thống, các logger khác được 
        # tạo ra đều kế thừa logger này 
        root_logger.setLevel(level)
        # Thiết lâp mức độ ghi logger thấp nhất cho logger này 
        # Thiết lập debug khi có nhu cầu coi sự thay đổi của hàm loss function 
        
        # Xóa handlers cũ nếu có
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Format logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # File handler (main log)
        file_handler = logging.FileHandler(
            self.main_log_path, 
            mode='w', 
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        return root_logger
    
    def get_module_logger(
        self, 
        module_name: str, 
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Lấy logger cho một module cụ thể với file log riêng
        
        Args:
            module_name: Tên module (e.g., 'data_integration', 'diagnostics')
            log_file: Tên file log (nếu None sẽ dùng {module_name}.log)
            level: Log level
        
        Returns:
            Logger instance cho module
        """
        logger = logging.getLogger(module_name)
        logger.setLevel(level)
        
        # Xóa handlers cũ
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Format logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # File handler
        if log_file is None:
            log_file = f'{module_name}.log'
        
        log_path = self.logs_dir / log_file
        
        file_handler = logging.FileHandler(
            log_path,
            mode='w',
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_logs_summary(self) -> dict:
        """Lấy thông tin tóm tắt về logs"""
        return {
            'logs_dir': str(self.logs_dir),
            'main_log': str(self.main_log_path),
            'module_logs': {
                'data_integration': str(self.data_integration_log_path),
                'diagnostics': str(self.diagnostics_log_path),
                'processing': str(self.processing_log_path),
            }
        }
