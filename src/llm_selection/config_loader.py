"""
配置文件加载器 - 支持JSON和YAML格式
"""
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import ValidationError

from llm_selection.models import ScenarioConfig


class ConfigLoadError(Exception):
    pass


def load_config_file(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)
    
    if not path.exists():
        raise ConfigLoadError(f"配置文件不存在: {file_path}")
    
    suffix = path.suffix.lower()
    
    try:
        if suffix in (".json",):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif suffix in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            raise ConfigLoadError(
                f"不支持的文件格式: {suffix}。支持的格式: .json, .yaml, .yml"
            )
    except json.JSONDecodeError as e:
        raise ConfigLoadError(f"JSON解析错误: {e}")
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"YAML解析错误: {e}")
    except UnicodeDecodeError as e:
        raise ConfigLoadError(f"文件编码错误: {e}")


def parse_scenario_config(data: Dict[str, Any]) -> ScenarioConfig:
    try:
        return ScenarioConfig(**data)
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            loc = " -> ".join(str(l) for l in error["loc"])
            msg = error["msg"]
            error_messages.append(f"[{loc}] {msg}")
        
        raise ConfigLoadError(
            "配置校验失败:\n" + "\n".join(f"  - {m}" for m in error_messages)
        )


def load_scenario_config(file_path: str) -> ScenarioConfig:
    data = load_config_file(file_path)
    return parse_scenario_config(data)
