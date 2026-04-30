"""
数据模型定义和校验逻辑
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationInfo,
)

from llm_selection.constants import PrivacyLevel


class PrimaryInferencePath(str, Enum):
    LOCAL = "local"
    API = "api"


class LatencyConstraintCheck(BaseModel):
    constraint_p95_ms: Optional[float] = Field(description="用户设置的P95延迟约束")
    estimated_p95_ms: float = Field(description="估算的P95延迟")
    violates_constraint: bool = Field(description="是否违反延迟约束")
    margin_ms: float = Field(description="与约束的差值（正值表示未违反，负值表示违反）")


class CostBreakdownLocal(BaseModel):
    hardware_amortization_monthly: float = Field(description="月度硬件摊销/租金")
    power_cost_monthly: float = Field(description="月度电费")
    total_monthly: float = Field(description="月度总成本")
    hardware_tier: str = Field(description="硬件层级名称")
    assumptions: Dict[str, Any] = Field(description="假设参数")


class CostBreakdownAPI(BaseModel):
    input_tokens_daily: int = Field(description="日均输入token数")
    output_tokens_daily: int = Field(description="日均输出token数")
    input_cost_monthly: float = Field(description="月度输入成本")
    output_cost_monthly: float = Field(description="月度输出成本")
    total_monthly: float = Field(description="月度总成本")
    pricing_model: str = Field(description="定价模型名称")
    assumptions: Dict[str, Any] = Field(description="假设参数")


class CostResult(BaseModel):
    local: CostBreakdownLocal = Field(description="本地部署月度成本分解")
    api: CostBreakdownAPI = Field(description="API部署月度成本分解")


class PrivacyFactors(BaseModel):
    data_leaves_premises: bool = Field(description="数据是否离境")
    third_party_logging: bool = Field(description="是否经过第三方日志留存")
    endpoint_is_external: bool = Field(description="端点是否为外部")


class PrivacyResult(BaseModel):
    local: PrivacyLevel
    api: PrivacyLevel
    factors_local: PrivacyFactors
    factors_api: PrivacyFactors
    rules_applied: Dict[str, str] = Field(description="应用的判定规则")


class LatencyBreakdown(BaseModel):
    mean_latency_ms: float = Field(description="平均延迟(ms)")
    p95_latency_ms: float = Field(description="P95延迟(ms)")
    constraint_check: Optional[LatencyConstraintCheck] = Field(
        default=None, description="与用户设置的P95约束的对比检查"
    )
    assumptions: Dict[str, Any] = Field(description="假设参数说明")


class LatencyResult(BaseModel):
    local: LatencyBreakdown = Field(description="本地部署延迟估算")
    api: LatencyBreakdown = Field(description="API部署延迟估算")


class CustomizationFactors(BaseModel):
    needs_finetuning: bool = Field(description="是否需要微调")
    can_self_host_weights: bool = Field(description="能否自托管权重")
    can_fix_output_schema: bool = Field(description="能否固定输出schema")


class CustomizationScore(BaseModel):
    total_score: float = Field(description="总得分 0-1")
    component_scores: Dict[str, float] = Field(description="各组件得分")
    weights_applied: Dict[str, float] = Field(description="应用的权重")


class CustomizationResult(BaseModel):
    local: CustomizationScore
    api: CustomizationScore
    factors_local: CustomizationFactors
    factors_api: CustomizationFactors


class ComparisonResult(BaseModel):
    scenario_name: str
    primary_inference_path: PrimaryInferencePath = Field(
        description="用户指定的主推理路径，用于摘要高亮或推荐结论"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    cost: CostResult
    privacy: PrivacyResult
    latency: LatencyResult
    customization: CustomizationResult
    input_config: Dict[str, Any] = Field(description="原始输入配置快照")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scenario_name": "example",
                    "primary_inference_path": "local",
                    "timestamp": "2026-04-30T12:00:00+00:00",
                    "cost": {
                        "local": {
                            "hardware_amortization_monthly": 800.0,
                            "power_cost_monthly": 69.12,
                            "total_monthly": 869.12,
                            "hardware_tier": "mid_gpu",
                            "assumptions": {}
                        },
                        "api": {
                            "input_tokens_daily": 1000000,
                            "output_tokens_daily": 500000,
                            "input_cost_monthly": 45.0,
                            "output_cost_monthly": 30.0,
                            "total_monthly": 75.0,
                            "pricing_model": "gpt_35_turbo",
                            "assumptions": {}
                        }
                    },
                    "privacy": {},
                    "latency": {},
                    "customization": {},
                    "input_config": {},
                }
            ]
        }
    }


class ScenarioConfig(BaseModel):
    scenario_name: str = Field(description="场景名称")
    
    daily_requests: int = Field(description="日均请求量量级")
    
    avg_input_tokens: int = Field(description="单次平均输入token数量")
    
    avg_output_tokens: int = Field(description="单次平均输出token数量")
    
    allow_external_network: bool = Field(
        description="是否允许连接外网"
    )
    
    latency_constraint_p95_ms: Optional[float] = Field(
        default=None, description="可接受的P95延迟约束(毫秒)"
    )
    
    needs_finetuning: bool = Field(
        default=False, description="是否存在微调硬性需求"
    )
    
    needs_strict_output_schema: bool = Field(
        default=False, description="是否存在强约束输出硬性需求"
    )
    
    primary_inference_path: PrimaryInferencePath = Field(
        description="主推理路径: local 或 api（仅用于摘要高亮，不影响两侧数据计算）"
    )
    
    data_leaves_premises_local: bool = Field(
        default=False, description="本地部署时数据是否离境"
    )
    
    third_party_logging_local: bool = Field(
        default=False, description="本地部署时是否有第三方日志留存"
    )
    
    api_pricing_model: str = Field(
        default="gpt_35_turbo", description="API定价模型名称"
    )
    
    custom_api_pricing: Optional[Dict[str, float]] = Field(
        default=None, description="自定义API定价: {input_per_1k_tokens, output_per_1k_tokens}"
    )
    
    @field_validator("scenario_name")
    @classmethod
    def validate_scenario_name(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("场景名称不能为空")
        return v
    
    @field_validator("avg_input_tokens", "avg_output_tokens")
    @classmethod
    def validate_non_negative_tokens(cls, v: int, info: ValidationInfo) -> int:
        if v < 0:
            raise ValueError(f"{info.field_name} 不能为负数")
        return v
    
    @field_validator("daily_requests")
    @classmethod
    def validate_positive_requests(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("日均请求量必须大于0")
        return v
    
    @field_validator("latency_constraint_p95_ms")
    @classmethod
    def validate_latency_constraint(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("延迟约束必须大于0")
        return v
    
    @model_validator(mode="after")
    def validate_mutually_exclusive_constraints(self) -> "ScenarioConfig":
        if (
            self.primary_inference_path == PrimaryInferencePath.API
            and not self.allow_external_network
        ):
            raise ValueError(
                "逻辑互斥: 禁止对外联网的情况下，不能将纯云端API作为主推理路径。"
                f"当前配置: primary_inference_path={self.primary_inference_path.value}, "
                f"allow_external_network={self.allow_external_network}"
            )
        
        if self.custom_api_pricing is not None:
            required_keys = {"input_per_1k_tokens", "output_per_1k_tokens"}
            if not required_keys.issubset(self.custom_api_pricing.keys()):
                raise ValueError(
                    f"custom_api_pricing 必须包含: {required_keys}"
                )
            for key in required_keys:
                if self.custom_api_pricing[key] < 0:
                    raise ValueError(f"custom_api_pricing.{key} 不能为负数")
        
        return self
