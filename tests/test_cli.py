"""
测试CLI接口
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import pytest
from click.testing import CliRunner

from llm_selection.cli import main


class TestCLIExitCodes:
    
    def test_invalid_config_exits_nonzero(self, create_temp_json):
        invalid_config = {
            "scenario_name": "test",
            "daily_requests": -100,
            "avg_input_tokens": 100,
            "avg_output_tokens": 50,
            "allow_external_network": True,
            "primary_inference_path": "local",
        }
        
        config_path = create_temp_json(invalid_config)
        
        runner = CliRunner()
        result = runner.invoke(main, [str(config_path)])
        
        assert result.exit_code != 0
        assert "配置校验失败" in result.output or "不能为负数" in result.output
    
    def test_mutually_exclusive_exits_nonzero(self, create_temp_json):
        invalid_config = {
            "scenario_name": "test",
            "daily_requests": 1000,
            "avg_input_tokens": 100,
            "avg_output_tokens": 50,
            "allow_external_network": False,
            "primary_inference_path": "api",
        }
        
        config_path = create_temp_json(invalid_config)
        
        runner = CliRunner()
        result = runner.invoke(main, [str(config_path)])
        
        assert result.exit_code != 0
        assert "逻辑互斥" in result.output or "禁止对外联网" in result.output
    
    def test_valid_config_exits_zero(self, create_temp_json, valid_local_scenario):
        config_path = create_temp_json(valid_local_scenario)
        
        runner = CliRunner()
        result = runner.invoke(main, [str(config_path)])
        
        assert result.exit_code == 0
    
    def test_nonexistent_file_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(main, ["/nonexistent/path/config.json"])
        
        assert result.exit_code != 0


class TestCLIJsonOutput:
    
    def test_json_only_output(self, create_temp_json, valid_local_scenario):
        config_path = create_temp_json(valid_local_scenario)
        
        runner = CliRunner()
        result = runner.invoke(main, [str(config_path), "--json-only"])
        
        assert result.exit_code == 0
        
        output_data = json.loads(result.output)
        assert output_data["scenario_name"] == "test_local_scenario"
        assert "cost" in output_data
        assert "privacy" in output_data
        assert "latency" in output_data
        assert "customization" in output_data
        assert "input_config" in output_data
    
    def test_json_pretty_output(self, create_temp_json, valid_local_scenario):
        config_path = create_temp_json(valid_local_scenario)
        
        runner = CliRunner()
        result = runner.invoke(main, [str(config_path), "--json-only", "--json-pretty"])
        
        assert result.exit_code == 0
        assert "\n" in result.output
        assert "  " in result.output or "\t" in result.output
        
        output_data = json.loads(result.output)
        assert output_data["scenario_name"] == "test_local_scenario"
    
    def test_json_output_matches_pydantic_model(self, create_temp_json, valid_api_scenario):
        config_path = create_temp_json(valid_api_scenario)
        
        runner = CliRunner()
        result = runner.invoke(main, [str(config_path), "--json-only"])
        
        assert result.exit_code == 0
        
        from llm_selection.models import ComparisonResult
        output_data = json.loads(result.output)
        
        validated_result = ComparisonResult(**output_data)
        assert validated_result.scenario_name == "test_api_scenario"
        assert validated_result.cost.api is not None


class TestCLIHelp:
    
    def test_help_contains_json_only_info(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        
        assert result.exit_code == 0
        assert "--json-only" in result.output
        assert "机器可读" in result.output
    
    def test_help_contains_behavior_description(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        
        assert result.exit_code == 0
        assert "默认行为" in result.output or "行为说明" in result.output


class TestCLISubprocess:
    
    def test_subprocess_invalid_config_exit_code(self, tmp_path: Path):
        config_path = tmp_path / "invalid.json"
        config_path.write_text(json.dumps({
            "scenario_name": "test",
            "daily_requests": -5,
            "avg_input_tokens": 100,
            "avg_output_tokens": 50,
            "allow_external_network": True,
            "primary_inference_path": "local",
        }))
        
        result = subprocess.run(
            [sys.executable, "-m", "llm_selection.cli", str(config_path)],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode != 0
        assert len(result.stderr) > 0 or "失败" in result.stdout
    
    def test_subprocess_valid_config_exit_code(self, tmp_path: Path, valid_local_scenario):
        config_path = tmp_path / "valid.json"
        config_path.write_text(json.dumps(valid_local_scenario))
        
        result = subprocess.run(
            [sys.executable, "-m", "llm_selection.cli", str(config_path), "--json-only"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        
        output_data = json.loads(result.stdout)
        assert output_data["scenario_name"] == "test_local_scenario"
