"""
常量定义和配置参数
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict


class PrivacyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


CUSTOMIZATION_WEIGHTS: Dict[str, float] = {
    "needs_finetuning": 0.4,
    "can_self_host_weights": 0.35,
    "can_fix_output_schema": 0.25,
}

LOCAL_HARDWARE_TIERS = [
    {
        "name": "entry_gpu",
        "min_requests_per_day": 0,
        "max_requests_per_day": 1000,
        "monthly_amortization": 300.0,
        "power_usage_kwh_per_hour": 0.3,
        "tokens_per_second": 50,
        "batch_size": 4,
    },
    {
        "name": "mid_gpu",
        "min_requests_per_day": 1001,
        "max_requests_per_day": 10000,
        "monthly_amortization": 800.0,
        "power_usage_kwh_per_hour": 0.8,
        "tokens_per_second": 120,
        "batch_size": 8,
    },
    {
        "name": "high_end_gpu",
        "min_requests_per_day": 10001,
        "max_requests_per_day": 100000,
        "monthly_amortization": 2000.0,
        "power_usage_kwh_per_hour": 1.5,
        "tokens_per_second": 300,
        "batch_size": 16,
    },
    {
        "name": "multi_gpu",
        "min_requests_per_day": 100001,
        "max_requests_per_day": 1000000,
        "monthly_amortization": 5000.0,
        "power_usage_kwh_per_hour": 4.0,
        "tokens_per_second": 800,
        "batch_size": 32,
    },
]

ELECTRICITY_COST_PER_KWH = 0.12

API_PRICING_DEFAULTS = {
    "gpt_4": {
        "input_per_1k_tokens": 0.03,
        "output_per_1k_tokens": 0.06,
    },
    "gpt_35_turbo": {
        "input_per_1k_tokens": 0.0015,
        "output_per_1k_tokens": 0.002,
    },
    "claude_3_opus": {
        "input_per_1k_tokens": 0.015,
        "output_per_1k_tokens": 0.075,
    },
    "claude_3_sonnet": {
        "input_per_1k_tokens": 0.003,
        "output_per_1k_tokens": 0.015,
    },
}

LATENCY_ASSUMPTIONS = {
    "local": {
        "mean_per_token_ms": 20,
        "p95_multiplier": 1.5,
        "is_assumption": True,
    },
    "api": {
        "rtt_ms": 100,
        "queue_wait_ms": 50,
        "mean_per_token_ms": 30,
        "p95_multiplier": 2.0,
        "is_assumption": True,
    },
}

DAYS_PER_MONTH = 30
HOURS_PER_DAY = 24
