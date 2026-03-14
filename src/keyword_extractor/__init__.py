"""
Keyword Extractor - 中文关键词提取工具

基于 KeyBERT 和 LLM 的中文关键词提取工具，针对中文文本优化。
"""

__version__ = "0.1.0"
__author__ = "Hao"

from .extractor import KeywordExtractor
from .bert_memory import BertMemoryExtractor
from .models import ExtractionResult, ExtractorConfig

__all__ = ["KeywordExtractor", "BertMemoryExtractor", "ExtractionResult", "ExtractorConfig"]
