"""
Pipeline Modules Package
========================

Sub-packages for the WorldBank Data Processing Pipeline:

Modules:
  - config_handler: Configuration management and validation
  - data_integration: Phase 0 - Data merging and integration
  - diagnostics: Phase 1 - Data profiling and analysis
  - processing: Phase 2 - Data preprocessing and transformation
"""

__version__ = '1.0.0'
__author__ = 'Data Team'

from . import config_handler
from . import data_integration
from . import diagnostics
from ...EDA import processing
from . import run_manager

__all__ = [
    'config_handler',
    'data_integration',
    'diagnostics',
    'processing',
]
