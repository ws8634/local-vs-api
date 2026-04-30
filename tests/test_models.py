"""
测试配置模型和校验逻辑
"""
import pytest
from pydantic import ValidationError

from llm_selection.models import (
    ScenarioConfig,
    PrimaryInferencePath,
    ComparisonResult,
    CostResult,
    CostBreakdownLocal,
    CostBreakdownAPI,
    LatencyResult,
    LatencyBreakdown,
    LatencyConstraintCheck,
    PrivacyResult,
    PrivacyFactors,
    CustomizationResult,
    CustomizationScore,
    CustomizationFactors,
)
from llm_selection.config_loader import parse_scenario_config, ConfigLoadError


class TestScenarioConfigValidation:
    
    def test_valid_local_config(self, valid_local_scenario):
        config = ScenarioConfig(**valid_local_scenario)
        assert config.scenario_name == "test_local_scenario"
        assert config.primary_inference_path == PrimaryInferencePath.LOCAL
    
    def test_valid_api_config(self, valid_api_scenario):
        config = ScenarioConfig(**valid_api_scenario)
        assert config.primary_inference_path == PrimaryInferencePath.API
    
    def test_negative_input_tokens(self, valid_local_scenario):
        valid_local_scenario["avg_input_tokens"] = -100
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(**valid_local_scenario)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("avg_input_tokens",)
        assert "不能为负数" in errors[0]["msg"] or "Value error" in errors[0]["type"]
    
    def test_negative_output_tokens(self, valid_local_scenario):
        valid_local_scenario["avg_output_tokens"] = -50
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(**valid_local_scenario)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("avg_output_tokens",)
    
    def test_zero_requests(self, valid_local_scenario):
        valid_local_scenario["daily_requests"] = 0
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(**valid_local_scenario)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("daily_requests",)
    
    def test_negative_requests(self, valid_local_scenario):
        valid_local_scenario["daily_requests"] = -100
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(**valid_local_scenario)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("daily_requests",)
    
    def test_mutually_exclusive_api_without_network(self, valid_api_scenario):
        valid_api_scenario["allow_external_network"] = False
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(**valid_api_scenario)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
    
    def test_custom_api_pricing_valid(self, valid_api_scenario):
        valid_api_scenario["custom_api_pricing"] = {
            "input_per_1k_tokens": 0.001,
            "output_per_1k_tokens": 0.002,
        }
        config = ScenarioConfig(**valid_api_scenario)
        assert config.custom_api_pricing is not None
        assert config.custom_api_pricing["input_per_1k_tokens"] == 0.001
    
    def test_custom_api_pricing_negative(self, valid_api_scenario):
        valid_api_scenario["custom_api_pricing"] = {
            "input_per_1k_tokens": -0.001,
            "output_per_1k_tokens": 0.002,
        }
        with pytest.raises(ValidationError):
            ScenarioConfig(**valid_api_scenario)
    
    def test_custom_api_pricing_missing_key(self, valid_api_scenario):
        valid_api_scenario["custom_api_pricing"] = {
            "input_per_1k_tokens": 0.001,
        }
        with pytest.raises(ValidationError):
            ScenarioConfig(**valid_api_scenario)
    
    def test_empty_scenario_name(self, valid_local_scenario):
        valid_local_scenario["scenario_name"] = ""
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(**valid_local_scenario)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("scenario_name",)
    
    def test_whitespace_only_scenario_name(self, valid_local_scenario):
        valid_local_scenario["scenario_name"] = "   "
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(**valid_local_scenario)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("scenario_name",)
    
    def test_latency_constraint_negative(self, valid_local_scenario):
        valid_local_scenario["latency_constraint_p95_ms"] = -100.0
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(**valid_local_scenario)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("latency_constraint_p95_ms",)
    
    def test_latency_constraint_zero(self, valid_local_scenario):
        valid_local_scenario["latency_constraint_p95_ms"] = 0.0
        with pytest.raises(ValidationError):
            ScenarioConfig(**valid_local_scenario)
    
    def test_latency_constraint_valid(self, valid_local_scenario):
        valid_local_scenario["latency_constraint_p95_ms"] = 5000.0
        config = ScenarioConfig(**valid_local_scenario)
        assert config.latency_constraint_p95_ms == 5000.0


class TestParseScenarioConfig:
    
    def test_parse_valid_config(self, valid_local_scenario):
        config = parse_scenario_config(valid_local_scenario)
        assert isinstance(config, ScenarioConfig)
    
    def test_parse_invalid_config_raises_config_load_error(self):
        invalid_data = {
            "scenario_name": "test",
            "daily_requests": -100,
            "avg_input_tokens": 100,
            "avg_output_tokens": 50,
            "allow_external_network": True,
            "primary_inference_path": "local",
        }
        with pytest.raises(ConfigLoadError):
            parse_scenario_config(invalid_data)
    
    def test_parse_with_wrong_enum_value(self, valid_local_scenario):
        valid_local_scenario["primary_inference_path"] = "invalid_value"
        with pytest.raises(ConfigLoadError):
            parse_scenario_config(valid_local_scenario)


class TestOutputModelStructure:
    
    def test_comparison_result_has_both_costs(self):
        from llm_selection.constants import PrivacyLevel
        
        privacy_result = PrivacyResult(
            local=PrivacyLevel.HIGH,
            api=PrivacyLevel.LOW,
            factors_local=PrivacyFactors(
                data_leaves_premises=False,
                third_party_logging=False,
                endpoint_is_external=False,
            ),
            factors_api=PrivacyFactors(
                data_leaves_premises=True,
                third_party_logging=True,
                endpoint_is_external=True,
            ),
            rules_applied={},
        )
        
        latency_result = LatencyResult(
            local=LatencyBreakdown(
                mean_latency_ms=1000.0,
                p95_latency_ms=1500.0,
                assumptions={},
            ),
            api=LatencyBreakdown(
                mean_latency_ms=2000.0,
                p95_latency_ms=3000.0,
                assumptions={},
            ),
        )
        
        customization_result = CustomizationResult(
            local=CustomizationScore(
                total_score=1.0,
                component_scores={},
                weights_applied={},
            ),
            api=CustomizationScore(
                total_score=0.6,
                component_scores={},
                weights_applied={},
            ),
            factors_local=CustomizationFactors(
                needs_finetuning=False,
                can_self_host_weights=True,
                can_fix_output_schema=True,
            ),
            factors_api=CustomizationFactors(
                needs_finetuning=False,
                can_self_host_weights=False,
                can_fix_output_schema=True,
            ),
        )
        
        result = ComparisonResult(
            scenario_name="test",
            primary_inference_path=PrimaryInferencePath.LOCAL,
            cost=CostResult(
                local=CostBreakdownLocal(
                    hardware_amortization_monthly=800.0,
                    power_cost_monthly=69.12,
                    total_monthly=869.12,
                    hardware_tier="mid_gpu",
                    assumptions={},
                ),
                api=CostBreakdownAPI(
                    input_tokens_daily=1000000,
                    output_tokens_daily=500000,
                    input_cost_monthly=45.0,
                    output_cost_monthly=30.0,
                    total_monthly=75.0,
                    pricing_model="gpt_35_turbo",
                    assumptions={},
                ),
            ),
            privacy=privacy_result,
            latency=latency_result,
            customization=customization_result,
            input_config={},
        )
        assert result.cost.local is not None
        assert result.cost.api is not None
    
    def test_latency_breakdown_has_constraint_check(self):
        constraint_check = LatencyConstraintCheck(
            constraint_p95_ms=2000.0,
            estimated_p95_ms=1500.0,
            violates_constraint=False,
            margin_ms=500.0,
        )
        
        breakdown = LatencyBreakdown(
            mean_latency_ms=1000.0,
            p95_latency_ms=1500.0,
            constraint_check=constraint_check,
            assumptions={},
        )
        
        assert breakdown.constraint_check is not None
        assert breakdown.constraint_check.violates_constraint is False
        assert breakdown.constraint_check.margin_ms == 500.0
    
    def test_latency_breakdown_without_constraint(self):
        breakdown = LatencyBreakdown(
            mean_latency_ms=1000.0,
            p95_latency_ms=1500.0,
            constraint_check=None,
            assumptions={},
        )
        
        assert breakdown.constraint_check is None
