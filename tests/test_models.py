"""
测试配置模型和校验逻辑
"""
import pytest
from pydantic import ValidationError

from llm_selection.models import ScenarioConfig, PrimaryInferencePath
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
        with pytest.raises(ValidationError, match="不能为负数"):
            ScenarioConfig(**valid_local_scenario)
    
    def test_negative_output_tokens(self, valid_local_scenario):
        valid_local_scenario["avg_output_tokens"] = -50
        with pytest.raises(ValidationError, match="不能为负数"):
            ScenarioConfig(**valid_local_scenario)
    
    def test_zero_requests(self, valid_local_scenario):
        valid_local_scenario["daily_requests"] = 0
        with pytest.raises(ValidationError, match="必须大于0"):
            ScenarioConfig(**valid_local_scenario)
    
    def test_negative_requests(self, valid_local_scenario):
        valid_local_scenario["daily_requests"] = -100
        with pytest.raises(ValidationError, match="必须大于0"):
            ScenarioConfig(**valid_local_scenario)
    
    def test_mutually_exclusive_api_without_network(self, valid_api_scenario):
        valid_api_scenario["allow_external_network"] = False
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(**valid_api_scenario)
        
        error_msg = str(exc_info.value)
        assert "逻辑互斥" in error_msg
        assert "禁止对外联网" in error_msg
        assert "纯云端API" in error_msg
    
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
        with pytest.raises(ValidationError, match="不能为负数"):
            ScenarioConfig(**valid_api_scenario)
    
    def test_custom_api_pricing_missing_key(self, valid_api_scenario):
        valid_api_scenario["custom_api_pricing"] = {
            "input_per_1k_tokens": 0.001,
        }
        with pytest.raises(ValidationError, match="必须包含"):
            ScenarioConfig(**valid_api_scenario)
    
    def test_empty_scenario_name(self, valid_local_scenario):
        valid_local_scenario["scenario_name"] = ""
        with pytest.raises(ValidationError, match="至少一个字符"):
            ScenarioConfig(**valid_local_scenario)
    
    def test_latency_constraint_negative(self, valid_local_scenario):
        valid_local_scenario["latency_constraint_p95_ms"] = -100.0
        with pytest.raises(ValidationError, match="大于0"):
            ScenarioConfig(**valid_local_scenario)


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
        with pytest.raises(ConfigLoadError, match="配置校验失败"):
            parse_scenario_config(invalid_data)
    
    def test_parse_with_wrong_enum_value(self, valid_local_scenario):
        valid_local_scenario["primary_inference_path"] = "invalid_value"
        with pytest.raises(ConfigLoadError):
            parse_scenario_config(valid_local_scenario)
