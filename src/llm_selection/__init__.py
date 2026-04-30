"""
LLM选型对比工具 - 本地模型 vs API模型
"""
from llm_selection.models import ScenarioConfig, ComparisonResult
from llm_selection.calculator import run_comparison

__version__ = "0.1.0"
__all__ = ["ScenarioConfig", "ComparisonResult", "run_comparison"]
