"""
四大维度计算引擎
"""
from typing import Dict, Any, Optional
import math

from llm_selection.models import (
    ScenarioConfig,
    ComparisonResult,
    CostResult,
    CostBreakdownLocal,
    CostBreakdownAPI,
    PrivacyResult,
    PrivacyFactors,
    LatencyResult,
    LatencyBreakdown,
    CustomizationResult,
    CustomizationScore,
    CustomizationFactors,
    PrimaryInferencePath,
)
from llm_selection.constants import (
    PrivacyLevel,
    CUSTOMIZATION_WEIGHTS,
    LOCAL_HARDWARE_TIERS,
    ELECTRICITY_COST_PER_KWH,
    API_PRICING_DEFAULTS,
    LATENCY_ASSUMPTIONS,
    DAYS_PER_MONTH,
    HOURS_PER_DAY,
)


def calculate_cost_local(config: ScenarioConfig) -> Optional[CostBreakdownLocal]:
    if config.primary_inference_path == PrimaryInferencePath.API:
        return None
    
    daily_requests = config.daily_requests
    
    selected_tier = None
    for tier in LOCAL_HARDWARE_TIERS:
        if tier["min_requests_per_day"] <= daily_requests <= tier["max_requests_per_day"]:
            selected_tier = tier
            break
    
    if selected_tier is None:
        selected_tier = LOCAL_HARDWARE_TIERS[-1]
    
    hardware_amortization = selected_tier["monthly_amortization"]
    
    daily_operational_hours = 24
    power_usage_hourly = selected_tier["power_usage_kwh_per_hour"]
    monthly_power_cost = (
        power_usage_hourly 
        * daily_operational_hours 
        * DAYS_PER_MONTH 
        * ELECTRICITY_COST_PER_KWH
    )
    
    total_monthly = hardware_amortization + monthly_power_cost
    
    return CostBreakdownLocal(
        hardware_amortization_monthly=hardware_amortization,
        power_cost_monthly=round(monthly_power_cost, 4),
        total_monthly=round(total_monthly, 4),
        hardware_tier=selected_tier["name"],
        assumptions={
            "electricity_cost_per_kwh": ELECTRICITY_COST_PER_KWH,
            "daily_operational_hours": daily_operational_hours,
            "hardware_tier_tokens_per_second": selected_tier["tokens_per_second"],
            "hardware_tier_batch_size": selected_tier["batch_size"],
            "note": "硬件摊销为阶跃函数，基于日均请求量选择对应层级",
        },
    )


def calculate_cost_api(config: ScenarioConfig) -> Optional[CostBreakdownAPI]:
    if config.primary_inference_path == PrimaryInferencePath.LOCAL:
        return None
    
    input_tokens_daily = config.daily_requests * config.avg_input_tokens
    output_tokens_daily = config.daily_requests * config.avg_output_tokens
    
    if config.custom_api_pricing is not None:
        pricing = config.custom_api_pricing
        pricing_model = "custom"
    else:
        pricing = API_PRICING_DEFAULTS.get(
            config.api_pricing_model,
            API_PRICING_DEFAULTS["gpt_35_turbo"],
        )
        pricing_model = config.api_pricing_model
    
    input_per_1k = pricing["input_per_1k_tokens"]
    output_per_1k = pricing["output_per_1k_tokens"]
    
    input_cost_monthly = (input_tokens_daily / 1000) * input_per_1k * DAYS_PER_MONTH
    output_cost_monthly = (output_tokens_daily / 1000) * output_per_1k * DAYS_PER_MONTH
    
    total_monthly = input_cost_monthly + output_cost_monthly
    
    return CostBreakdownAPI(
        input_tokens_daily=input_tokens_daily,
        output_tokens_daily=output_tokens_daily,
        input_cost_monthly=round(input_cost_monthly, 4),
        output_cost_monthly=round(output_cost_monthly, 4),
        total_monthly=round(total_monthly, 4),
        pricing_model=pricing_model,
        assumptions={
            "input_per_1k_tokens": input_per_1k,
            "output_per_1k_tokens": output_per_1k,
            "days_per_month": DAYS_PER_MONTH,
            "note": "API成本为线性计算，按实际token用量计费",
        },
    )


def calculate_cost(config: ScenarioConfig) -> CostResult:
    return CostResult(
        local=calculate_cost_local(config),
        api=calculate_cost_api(config),
    )


def calculate_privacy(config: ScenarioConfig) -> PrivacyResult:
    factors_local = PrivacyFactors(
        data_leaves_premises=config.data_leaves_premises_local,
        third_party_logging=config.third_party_logging_local,
        endpoint_is_external=False,
    )
    
    factors_api = PrivacyFactors(
        data_leaves_premises=True,
        third_party_logging=True,
        endpoint_is_external=True,
    )
    
    def determine_level(factors: PrivacyFactors) -> PrivacyLevel:
        risk_score = 0
        if factors.data_leaves_premises:
            risk_score += 2
        if factors.third_party_logging:
            risk_score += 2
        if factors.endpoint_is_external:
            risk_score += 1
        
        if risk_score == 0:
            return PrivacyLevel.HIGH
        elif risk_score <= 2:
            return PrivacyLevel.MEDIUM
        else:
            return PrivacyLevel.LOW
    
    level_local = determine_level(factors_local)
    level_api = determine_level(factors_api)
    
    rules_applied = {
        "local_level": f"基于本地隐私因子判定为 {level_local.value}",
        "api_level": f"基于API隐私因子判定为 {level_api.value}",
        "rule_data_leaves": "数据离境=高风险(+2分)",
        "rule_third_party": "第三方日志=高风险(+2分)",
        "rule_external_endpoint": "外部端点=中等风险(+1分)",
        "score_interpretation": "0分=高隐私等级, 1-2分=中等等级, 3+分=低等级",
    }
    
    return PrivacyResult(
        local=level_local,
        api=level_api,
        factors_local=factors_local,
        factors_api=factors_api,
        rules_applied=rules_applied,
    )


def calculate_latency(config: ScenarioConfig) -> LatencyResult:
    total_tokens_per_request = config.avg_input_tokens + config.avg_output_tokens
    
    local_breakdown: Optional[LatencyBreakdown] = None
    if config.primary_inference_path == PrimaryInferencePath.LOCAL:
        local_assumptions = LATENCY_ASSUMPTIONS["local"]
        mean_per_token = local_assumptions["mean_per_token_ms"]
        p95_multiplier = local_assumptions["p95_multiplier"]
        
        mean_latency = total_tokens_per_request * mean_per_token
        p95_latency = mean_latency * p95_multiplier
        
        local_breakdown = LatencyBreakdown(
            mean_latency_ms=round(mean_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            assumptions={
                "mean_per_token_ms": mean_per_token,
                "p95_multiplier": p95_multiplier,
                "total_tokens_per_request": total_tokens_per_request,
                "is_assumption": True,
                "note": "本地延迟基于每token处理时间估算，不考虑网络开销",
            },
        )
    
    api_breakdown: Optional[LatencyBreakdown] = None
    if config.primary_inference_path == PrimaryInferencePath.API:
        api_assumptions = LATENCY_ASSUMPTIONS["api"]
        rtt = api_assumptions["rtt_ms"]
        queue_wait = api_assumptions["queue_wait_ms"]
        mean_per_token = api_assumptions["mean_per_token_ms"]
        p95_multiplier = api_assumptions["p95_multiplier"]
        
        base_overhead = rtt + queue_wait
        token_processing = total_tokens_per_request * mean_per_token
        mean_latency = base_overhead + token_processing
        p95_latency = base_overhead + (token_processing * p95_multiplier)
        
        api_breakdown = LatencyBreakdown(
            mean_latency_ms=round(mean_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            assumptions={
                "rtt_ms": rtt,
                "queue_wait_ms": queue_wait,
                "mean_per_token_ms": mean_per_token,
                "p95_multiplier": p95_multiplier,
                "total_tokens_per_request": total_tokens_per_request,
                "is_assumption": True,
                "note": "API延迟包含网络往返时间、排队等待和token处理时间",
            },
        )
    
    return LatencyResult(
        local=local_breakdown,
        api=api_breakdown,
    )


def calculate_customization(config: ScenarioConfig) -> CustomizationResult:
    needs_finetuning = config.needs_finetuning
    needs_strict_schema = config.needs_strict_output_schema
    
    factors_local = CustomizationFactors(
        needs_finetuning=needs_finetuning,
        can_self_host_weights=True,
        can_fix_output_schema=True,
    )
    
    factors_api = CustomizationFactors(
        needs_finetuning=needs_finetuning,
        can_self_host_weights=False,
        can_fix_output_schema=needs_strict_schema,
    )
    
    def calculate_score(factors: CustomizationFactors) -> CustomizationScore:
        component_scores: Dict[str, float] = {}
        
        if factors.needs_finetuning:
            component_scores["needs_finetuning"] = (
                1.0 if factors.can_self_host_weights else 0.0
            )
        else:
            component_scores["needs_finetuning"] = 1.0
        
        component_scores["can_self_host_weights"] = (
            1.0 if factors.can_self_host_weights else 0.0
        )
        component_scores["can_fix_output_schema"] = (
            1.0 if factors.can_fix_output_schema else 0.0
        )
        
        total_score = sum(
            component_scores[key] * CUSTOMIZATION_WEIGHTS[key]
            for key in CUSTOMIZATION_WEIGHTS
        )
        
        return CustomizationScore(
            total_score=round(total_score, 4),
            component_scores=component_scores,
            weights_applied=CUSTOMIZATION_WEIGHTS.copy(),
        )
    
    score_local = calculate_score(factors_local)
    score_api = calculate_score(factors_api)
    
    return CustomizationResult(
        local=score_local,
        api=score_api,
        factors_local=factors_local,
        factors_api=factors_api,
    )


def run_comparison(config: ScenarioConfig) -> ComparisonResult:
    return ComparisonResult(
        scenario_name=config.scenario_name,
        cost=calculate_cost(config),
        privacy=calculate_privacy(config),
        latency=calculate_latency(config),
        customization=calculate_customization(config),
        input_config=config.model_dump(mode="json"),
    )
