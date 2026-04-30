"""
pytest fixtures
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator

import pytest

from llm_selection.models import PrimaryInferencePath


@pytest.fixture
def valid_local_scenario() -> Dict[str, Any]:
    return {
        "scenario_name": "test_local_scenario",
        "daily_requests": 5000,
        "avg_input_tokens": 200,
        "avg_output_tokens": 150,
        "allow_external_network": True,
        "latency_constraint_p95_ms": 2000.0,
        "needs_finetuning": False,
        "needs_strict_output_schema": False,
        "primary_inference_path": PrimaryInferencePath.LOCAL.value,
        "data_leaves_premises_local": False,
        "third_party_logging_local": False,
    }


@pytest.fixture
def valid_api_scenario() -> Dict[str, Any]:
    return {
        "scenario_name": "test_api_scenario",
        "daily_requests": 10000,
        "avg_input_tokens": 100,
        "avg_output_tokens": 50,
        "allow_external_network": True,
        "latency_constraint_p95_ms": 5000.0,
        "needs_finetuning": False,
        "needs_strict_output_schema": True,
        "primary_inference_path": PrimaryInferencePath.API.value,
        "api_pricing_model": "gpt_35_turbo",
    }


@pytest.fixture
def temp_json_file() -> Generator[Path, None, None]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)
    
    yield temp_path
    
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def create_temp_json():
    def _create_temp_json(data: Dict[str, Any]) -> Path:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)
        return temp_path
    
    return _create_temp_json


@pytest.fixture
def create_temp_yaml():
    def _create_temp_yaml(data: Dict[str, Any]) -> Path:
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            temp_path = Path(f.name)
        return temp_path
    
    return _create_temp_yaml
