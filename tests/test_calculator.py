"""
测试四大维度计算引擎
数值快照测试 - 在固定fixture下对比关键数字
"""
import pytest

from llm_selection.models import (
    ScenarioConfig,
    PrimaryInferencePath,
    LatencyConstraintCheck,
)
from llm_selection.calculator import (
    run_comparison,
    calculate_cost_local,
    calculate_cost_api,
    calculate_privacy,
    calculate_latency,
    calculate_customization,
    build_latency_constraint_check,
)
from llm_selection.constants import (
    PrivacyLevel,
    CUSTOMIZATION_WEIGHTS,
    LOCAL_HARDWARE_TIERS,
    ELECTRICITY_COST_PER_KWH,
    DAYS_PER_MONTH,
)


class TestCostCalculation:
    
    def test_local_cost_entry_tier(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=500,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
        )
        
        breakdown = calculate_cost_local(config)
        assert breakdown is not None
        
        expected_tier = LOCAL_HARDWARE_TIERS[0]
        assert breakdown.hardware_tier == expected_tier["name"]
        assert breakdown.hardware_amortization_monthly == pytest.approx(expected_tier["monthly_amortization"])
        
        expected_power_cost = (
            expected_tier["power_usage_kwh_per_hour"]
            * 24
            * DAYS_PER_MONTH
            * ELECTRICITY_COST_PER_KWH
        )
        assert breakdown.power_cost_monthly == pytest.approx(expected_power_cost, rel=1e-4)
        assert breakdown.total_monthly == pytest.approx(
            expected_tier["monthly_amortization"] + expected_power_cost,
            rel=1e-4
        )
    
    def test_local_cost_mid_tier(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=5000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
        )
        
        breakdown = calculate_cost_local(config)
        assert breakdown is not None
        assert breakdown.hardware_tier == LOCAL_HARDWARE_TIERS[1]["name"]
    
    def test_api_cost_calculation(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=200,
            avg_output_tokens=100,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.API,
            api_pricing_model="gpt_35_turbo",
        )
        
        breakdown = calculate_cost_api(config)
        assert breakdown is not None
        
        expected_input_tokens_daily = 1000 * 200
        expected_output_tokens_daily = 1000 * 100
        assert breakdown.input_tokens_daily == expected_input_tokens_daily
        assert breakdown.output_tokens_daily == expected_output_tokens_daily
        
        input_per_1k = 0.0015
        output_per_1k = 0.002
        expected_input_cost = (expected_input_tokens_daily / 1000) * input_per_1k * DAYS_PER_MONTH
        expected_output_cost = (expected_output_tokens_daily / 1000) * output_per_1k * DAYS_PER_MONTH
        
        assert breakdown.input_cost_monthly == pytest.approx(expected_input_cost, rel=1e-4)
        assert breakdown.output_cost_monthly == pytest.approx(expected_output_cost, rel=1e-4)
        assert breakdown.total_monthly == pytest.approx(expected_input_cost + expected_output_cost, rel=1e-4)
    
    def test_both_costs_calculated_regardless_of_primary_path(self):
        config_local = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
        )
        
        config_api = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.API,
        )
        
        cost_local_local = calculate_cost_local(config_local)
        cost_api_local = calculate_cost_api(config_local)
        cost_local_api = calculate_cost_local(config_api)
        cost_api_api = calculate_cost_api(config_api)
        
        assert cost_local_local is not None
        assert cost_api_local is not None
        assert cost_local_api is not None
        assert cost_api_api is not None
        
        assert cost_local_local.hardware_tier == cost_local_api.hardware_tier
        assert cost_api_local.total_monthly == pytest.approx(cost_api_api.total_monthly, rel=1e-4)


class TestPrivacyCalculation:
    
    def test_local_high_privacy(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
            data_leaves_premises_local=False,
            third_party_logging_local=False,
        )
        
        result = calculate_privacy(config)
        assert result.local == PrivacyLevel.HIGH
        assert result.api == PrivacyLevel.LOW
    
    def test_local_medium_privacy_data_leaves(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
            data_leaves_premises_local=True,
            third_party_logging_local=False,
        )
        
        result = calculate_privacy(config)
        assert result.local == PrivacyLevel.MEDIUM
    
    def test_local_low_privacy_both_risks(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
            data_leaves_premises_local=True,
            third_party_logging_local=True,
        )
        
        result = calculate_privacy(config)
        assert result.local == PrivacyLevel.LOW
    
    def test_privacy_factors_correct(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
        )
        
        result = calculate_privacy(config)
        
        assert result.factors_local.endpoint_is_external is False
        assert result.factors_api.endpoint_is_external is True
        assert result.factors_api.data_leaves_premises is True
        assert result.factors_api.third_party_logging is True
        
        assert "rule_data_leaves" in result.rules_applied
        assert "score_interpretation" in result.rules_applied


class TestLatencyConstraintCheck:
    
    def test_build_constraint_check_no_constraint(self):
        result = build_latency_constraint_check(None, 1000.0)
        assert result is None
    
    def test_build_constraint_check_not_violated(self):
        constraint = 2000.0
        estimated = 1500.0
        
        result = build_latency_constraint_check(constraint, estimated)
        
        assert result is not None
        assert result.constraint_p95_ms == constraint
        assert result.estimated_p95_ms == estimated
        assert result.violates_constraint is False
        assert result.margin_ms == pytest.approx(500.0)
    
    def test_build_constraint_check_violated(self):
        constraint = 1000.0
        estimated = 1500.0
        
        result = build_latency_constraint_check(constraint, estimated)
        
        assert result is not None
        assert result.violates_constraint is True
        assert result.margin_ms == pytest.approx(-500.0)
    
    def test_build_constraint_check_exact_match(self):
        constraint = 1000.0
        estimated = 1000.0
        
        result = build_latency_constraint_check(constraint, estimated)
        
        assert result is not None
        assert result.violates_constraint is False
        assert result.margin_ms == pytest.approx(0.0)


class TestLatencyCalculation:
    
    def test_local_and_api_latency_both_calculated(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
        )
        
        result = calculate_latency(config)
        
        assert result.local is not None
        assert result.api is not None
        
        assert result.local.mean_latency_ms > 0
        assert result.local.p95_latency_ms > 0
        assert result.api.mean_latency_ms > 0
        assert result.api.p95_latency_ms > 0
    
    def test_local_latency_calculation_values(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
        )
        
        result = calculate_latency(config)
        
        total_tokens = 150
        expected_mean = total_tokens * 20
        expected_p95 = expected_mean * 1.5
        
        assert result.local.mean_latency_ms == pytest.approx(expected_mean, rel=1e-2)
        assert result.local.p95_latency_ms == pytest.approx(expected_p95, rel=1e-2)
        assert result.local.assumptions["is_assumption"] is True
        assert result.local.constraint_check is None
    
    def test_api_latency_calculation_values(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.API,
        )
        
        result = calculate_latency(config)
        
        total_tokens = 150
        rtt = 100
        queue = 50
        per_token = 30
        
        base_overhead = rtt + queue
        token_processing = total_tokens * per_token
        expected_mean = base_overhead + token_processing
        expected_p95 = base_overhead + (token_processing * 2.0)
        
        assert result.api.mean_latency_ms == pytest.approx(expected_mean, rel=1e-2)
        assert result.api.p95_latency_ms == pytest.approx(expected_p95, rel=1e-2)
        assert result.api.assumptions["is_assumption"] is True
    
    def test_latency_with_constraint_not_violated(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
            latency_constraint_p95_ms=10000.0,
        )
        
        result = calculate_latency(config)
        
        assert result.local.constraint_check is not None
        assert result.api.constraint_check is not None
        
        assert result.local.constraint_check.violates_constraint is False
        assert result.local.constraint_check.margin_ms > 0
        
        assert result.api.constraint_check.violates_constraint is False
        assert result.api.constraint_check.margin_ms > 0
    
    def test_latency_with_constraint_violated(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=1000,
            avg_output_tokens=500,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
            latency_constraint_p95_ms=100.0,
        )
        
        result = calculate_latency(config)
        
        assert result.local.constraint_check is not None
        assert result.api.constraint_check is not None
        
        assert result.local.constraint_check.violates_constraint is True
        assert result.local.constraint_check.margin_ms < 0
        
        assert result.api.constraint_check.violates_constraint is True
        assert result.api.constraint_check.margin_ms < 0


class TestCustomizationCalculation:
    
    def test_local_always_full_score(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
            needs_finetuning=True,
            needs_strict_output_schema=True,
        )
        
        result = calculate_customization(config)
        
        assert result.local.total_score == pytest.approx(1.0, rel=1e-4)
        assert result.local.component_scores["needs_finetuning"] == 1.0
        assert result.local.component_scores["can_self_host_weights"] == 1.0
        assert result.local.component_scores["can_fix_output_schema"] == 1.0
    
    def test_api_score_without_finetuning_need(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.API,
            needs_finetuning=False,
            needs_strict_output_schema=True,
        )
        
        result = calculate_customization(config)
        
        expected_score = (
            1.0 * CUSTOMIZATION_WEIGHTS["needs_finetuning"]
            + 0.0 * CUSTOMIZATION_WEIGHTS["can_self_host_weights"]
            + 1.0 * CUSTOMIZATION_WEIGHTS["can_fix_output_schema"]
        )
        assert result.api.total_score == pytest.approx(expected_score, rel=1e-4)
    
    def test_api_score_with_finetuning_need(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.API,
            needs_finetuning=True,
            needs_strict_output_schema=True,
        )
        
        result = calculate_customization(config)
        
        expected_score = (
            0.0 * CUSTOMIZATION_WEIGHTS["needs_finetuning"]
            + 0.0 * CUSTOMIZATION_WEIGHTS["can_self_host_weights"]
            + 1.0 * CUSTOMIZATION_WEIGHTS["can_fix_output_schema"]
        )
        assert result.api.total_score == pytest.approx(expected_score, rel=1e-4)
    
    def test_weights_applied_in_result(self):
        config = ScenarioConfig(
            scenario_name="test",
            daily_requests=1000,
            avg_input_tokens=100,
            avg_output_tokens=50,
            allow_external_network=True,
            primary_inference_path=PrimaryInferencePath.LOCAL,
        )
        
        result = calculate_customization(config)
        
        assert result.local.weights_applied == CUSTOMIZATION_WEIGHTS
        assert result.api.weights_applied == CUSTOMIZATION_WEIGHTS


class TestFullComparisonSnapshot:
    
    def test_full_comparison_local_scenario_snapshot(self, valid_local_scenario):
        config = ScenarioConfig(**valid_local_scenario)
        result = run_comparison(config)
        
        assert result.scenario_name == "test_local_scenario"
        assert result.primary_inference_path == PrimaryInferencePath.LOCAL
        assert result.cost.local is not None
        assert result.cost.api is not None
        
        mid_tier = LOCAL_HARDWARE_TIERS[1]
        assert result.cost.local.hardware_tier == mid_tier["name"]
        
        expected_power_cost = (
            mid_tier["power_usage_kwh_per_hour"]
            * 24
            * DAYS_PER_MONTH
            * ELECTRICITY_COST_PER_KWH
        )
        expected_total = mid_tier["monthly_amortization"] + expected_power_cost
        
        assert result.cost.local.total_monthly == pytest.approx(expected_total, rel=1e-4)
        
        assert result.privacy.local == PrivacyLevel.HIGH
        assert result.privacy.api == PrivacyLevel.LOW
        
        assert result.latency.local is not None
        assert result.latency.api is not None
        
        total_tokens = 200 + 150
        expected_mean_local = total_tokens * 20
        assert result.latency.local.mean_latency_ms == pytest.approx(expected_mean_local, rel=1e-2)
        
        assert result.customization.local.total_score == pytest.approx(1.0, rel=1e-4)
        
        assert result.latency.local.constraint_check is not None
        assert result.latency.api.constraint_check is not None
    
    def test_full_comparison_api_scenario_snapshot(self, valid_api_scenario):
        config = ScenarioConfig(**valid_api_scenario)
        result = run_comparison(config)
        
        assert result.scenario_name == "test_api_scenario"
        assert result.primary_inference_path == PrimaryInferencePath.API
        assert result.cost.api is not None
        assert result.cost.local is not None
        
        daily_reqs = 10000
        input_tokens = 100
        output_tokens = 50
        
        expected_input_daily = daily_reqs * input_tokens
        expected_output_daily = daily_reqs * output_tokens
        assert result.cost.api.input_tokens_daily == expected_input_daily
        assert result.cost.api.output_tokens_daily == expected_output_daily
        
        assert result.privacy.api == PrivacyLevel.LOW
        
        assert result.latency.api is not None
        assert result.latency.local is not None
        
        total_tokens = input_tokens + output_tokens
        rtt = 100
        queue = 50
        per_token = 30
        expected_mean_api = rtt + queue + (total_tokens * per_token)
        assert result.latency.api.mean_latency_ms == pytest.approx(expected_mean_api, rel=1e-2)
        
        assert result.customization.api.component_scores["can_self_host_weights"] == 0.0
