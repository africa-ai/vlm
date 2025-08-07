"""
Visual Language Model Framework for Kalenjin Dictionary Processing
"""

__version__ = "1.0.0"
__author__ = "Dictionary Processing Team"

try:
    from .main import VLMProcessor
    from .config import VLMConfig
    __all__ = ['VLMProcessor', 'VLMConfig']
except ImportError:
    # Some dependencies may not be available
    __all__ = []
