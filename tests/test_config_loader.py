"""
测试配置文件加载器
"""
import json
from pathlib import Path

import pytest
import yaml

from llm_selection.config_loader import (
    load_config_file,
    parse_scenario_config,
    load_scenario_config,
    ConfigLoadError,
)


class TestLoadConfigFile:
    
    def test_load_valid_json(self, tmp_path: Path, valid_local_scenario):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(valid_local_scenario))
        
        data = load_config_file(str(config_path))
        assert data["scenario_name"] == "test_local_scenario"
    
    def test_load_valid_yaml(self, tmp_path: Path, valid_local_scenario):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(valid_local_scenario))
        
        data = load_config_file(str(config_path))
        assert data["scenario_name"] == "test_local_scenario"
    
    def test_load_valid_yml(self, tmp_path: Path, valid_local_scenario):
        config_path = tmp_path / "config.yml"
        config_path.write_text(yaml.dump(valid_local_scenario))
        
        data = load_config_file(str(config_path))
        assert data["scenario_name"] == "test_local_scenario"
    
    def test_nonexistent_file_raises(self):
        with pytest.raises(ConfigLoadError, match="配置文件不存在"):
            load_config_file("/nonexistent/path.json")
    
    def test_unsupported_extension_raises(self, tmp_path: Path):
        config_path = tmp_path / "config.txt"
        config_path.write_text("{}")
        
        with pytest.raises(ConfigLoadError, match="不支持的文件格式"):
            load_config_file(str(config_path))
    
    def test_invalid_json_raises(self, tmp_path: Path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{invalid json")
        
        with pytest.raises(ConfigLoadError, match="JSON解析错误"):
            load_config_file(str(config_path))
    
    def test_invalid_yaml_raises(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(": invalid\n  yaml: :")
        
        with pytest.raises(ConfigLoadError, match="YAML解析错误"):
            load_config_file(str(config_path))


class TestLoadScenarioConfig:
    
    def test_load_valid_scenario(self, tmp_path: Path, valid_local_scenario):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(valid_local_scenario))
        
        config = load_scenario_config(str(config_path))
        assert config.scenario_name == "test_local_scenario"
        assert config.daily_requests == 5000
    
    def test_load_invalid_scenario_raises(self, tmp_path: Path):
        invalid_data = {
            "scenario_name": "test",
            "daily_requests": -100,
            "avg_input_tokens": 100,
            "avg_output_tokens": 50,
            "allow_external_network": True,
            "primary_inference_path": "local",
        }
        
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(invalid_data))
        
        with pytest.raises(ConfigLoadError, match="配置校验失败"):
            load_scenario_config(str(config_path))
