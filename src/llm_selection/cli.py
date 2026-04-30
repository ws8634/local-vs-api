"""
命令行接口
"""
import json
import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from llm_selection import __version__
from llm_selection.config_loader import load_scenario_config, ConfigLoadError
from llm_selection.calculator import run_comparison
from llm_selection.models import ComparisonResult, PrimaryInferencePath


error_console = Console(stderr=True, style="bold red")
console = Console()


def format_cost_summary(result: ComparisonResult) -> str:
    lines = []
    
    if result.cost.local:
        local = result.cost.local
        lines.extend([
            f"[bold cyan]本地部署成本:[/bold cyan]",
            f"  硬件层级: {local.hardware_tier}",
            f"  月度硬件摊销: ${local.hardware_amortization_monthly:.2f}",
            f"  月度电费: ${local.power_cost_monthly:.2f}",
            f"  [bold]月度总成本: ${local.total_monthly:.2f}[/bold]",
        ])
    
    if result.cost.api:
        api = result.cost.api
        lines.extend([
            "",
            f"[bold cyan]API成本:[/bold cyan]",
            f"  定价模型: {api.pricing_model}",
            f"  日均输入tokens: {api.input_tokens_daily:,}",
            f"  日均输出tokens: {api.output_tokens_daily:,}",
            f"  月度输入成本: ${api.input_cost_monthly:.2f}",
            f"  月度输出成本: ${api.output_cost_monthly:.2f}",
            f"  [bold]月度总成本: ${api.total_monthly:.2f}[/bold]",
        ])
    
    return "\n".join(lines)


def format_privacy_summary(result: ComparisonResult) -> str:
    privacy = result.privacy
    lines = [
        f"[bold cyan]隐私等级:[/bold cyan]",
        f"  本地部署: [bold]{'green' if privacy.local.value == 'high' else 'yellow' if privacy.local.value == 'medium' else 'red'}]{privacy.local.value.upper()}[/bold]",
        f"  API部署: [bold]{'green' if privacy.api.value == 'high' else 'yellow' if privacy.api.value == 'medium' else 'red'}]{privacy.api.value.upper()}[/bold]",
    ]
    
    factors_local = privacy.factors_local
    factors_api = privacy.factors_api
    
    lines.extend([
        "",
        "[bold]本地隐私因子:[/bold]",
        f"  数据离境: {'是(高风险)' if factors_local.data_leaves_premises else '否'}",
        f"  第三方日志: {'是(高风险)' if factors_local.third_party_logging else '否'}",
        f"  外部端点: {'是(中风险)' if factors_local.endpoint_is_external else '否'}",
        "",
        "[bold]API隐私因子:[/bold]",
        f"  数据离境: {'是(高风险)' if factors_api.data_leaves_premises else '否'}",
        f"  第三方日志: {'是(高风险)' if factors_api.third_party_logging else '否'}",
        f"  外部端点: {'是(中风险)' if factors_api.endpoint_is_external else '否'}",
    ])
    
    return "\n".join(lines)


def format_latency_summary(result: ComparisonResult) -> str:
    lines = [f"[bold cyan]延迟估算:[/bold cyan]"]
    
    if result.latency.local:
        local = result.latency.local
        lines.extend([
            f"  本地部署:",
            f"    平均延迟: {local.mean_latency_ms:.1f} ms",
            f"    P95延迟: {local.p95_latency_ms:.1f} ms",
            f"    [dim italic]注: 以上为假设数据，基于每token处理时间估算[/dim italic]",
        ])
    
    if result.latency.api:
        api = result.latency.api
        if result.latency.local:
            lines.append("")
        lines.extend([
            f"  API部署:",
            f"    平均延迟: {api.mean_latency_ms:.1f} ms",
            f"    P95延迟: {api.p95_latency_ms:.1f} ms",
            f"    [dim italic]注: 以上为假设数据，包含网络RTT和排队等待[/dim italic]",
        ])
    
    return "\n".join(lines)


def format_customization_summary(result: ComparisonResult) -> str:
    custom = result.customization
    lines = [
        f"[bold cyan]定制化能力得分 (0-1):[/bold cyan]",
        f"  本地部署: [bold green]{custom.local.total_score:.2f}[/bold green]",
        f"  API部署: [bold]{'green' if custom.api.total_score >= 0.7 else 'yellow' if custom.api.total_score >= 0.4 else 'red'}]{custom.api.total_score:.2f}[/bold]",
    ]
    
    weights = custom.local.weights_applied
    lines.extend([
        "",
        "[bold]权重配置:[/bold]",
        f"  需要微调: {weights['needs_finetuning']:.0%}",
        f"  自托管权重: {weights['can_self_host_weights']:.0%}",
        f"  固定输出Schema: {weights['can_fix_output_schema']:.0%}",
    ])
    
    return "\n".join(lines)


def print_human_readable_summary(result: ComparisonResult):
    primary_path = result.input_config.get("primary_inference_path", "unknown")
    path_label = "本地部署" if primary_path == PrimaryInferencePath.LOCAL else "API部署"
    
    console.print(Panel.fit(
        f"[bold green]LLM选型对比结果[/bold green]\n"
        f"场景: [cyan]{result.scenario_name}[/cyan]\n"
        f"主推理路径: [yellow]{path_label}[/yellow]\n"
        f"时间: {result.timestamp}",
        title="对比摘要",
        border_style="green",
    ))
    
    console.print("")
    console.print(format_cost_summary(result))
    
    console.print("")
    console.print(format_privacy_summary(result))
    
    console.print("")
    console.print(format_latency_summary(result))
    
    console.print("")
    console.print(format_customization_summary(result))
    
    console.print("")
    console.print("[dim italic]提示: 使用 --json-only 标志获取完整机器可读输出[/dim italic]")


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--json-only",
    is_flag=True,
    default=False,
    help="仅输出机器可读的JSON格式结果到stdout，不打印人类可读摘要",
)
@click.option(
    "--json-pretty",
    is_flag=True,
    default=False,
    help="以美化缩进格式输出JSON（仅在--json-only时生效）",
)
@click.version_option(version=__version__)
def main(config_file: str, json_only: bool, json_pretty: bool):
    """
    LLM选型对比工具 - 对比本地模型与API模型在成本、隐私、延迟、定制化四个维度的表现。
    
    CONFIG_FILE: 场景配置文件路径，支持JSON或YAML格式。
    
    配置文件示例(scenario.json):
    {
        "scenario_name": "内部客服机器人",
        "daily_requests": 5000,
        "avg_input_tokens": 200,
        "avg_output_tokens": 150,
        "allow_external_network": true,
        "primary_inference_path": "local"
    }
    
    行为说明:
    - 默认行为: 打印人类可读摘要到控制台
    - 使用 --json-only: 仅输出完整JSON结果到stdout（便于管道处理）
    - 校验失败: 错误信息输出到stderr，退出码非零
    """
    try:
        config = load_scenario_config(config_file)
    except ConfigLoadError as e:
        error_console.print(f"配置加载失败: {e}")
        sys.exit(1)
    
    result = run_comparison(config)
    
    if json_only:
        indent = 2 if json_pretty else None
        json_str = result.model_dump_json(indent=indent)
        click.echo(json_str)
    else:
        print_human_readable_summary(result)
        
        click.echo("\n" + "=" * 60)
        click.echo("机器可读JSON输出:")
        click.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
