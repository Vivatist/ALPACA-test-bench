"""
Инициализация основного модуля ALPACA Test Bench.
"""

from .core.pipeline import DocumentPipeline
from .core.base import *
from .utils import *

__version__ = "1.0.0"
__author__ = "ALPACA Team"
__email__ = "team@alpaca.dev"

__all__ = [
    'DocumentPipeline',
]