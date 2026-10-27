#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents import get_agent
from minisweagent.agents.utils.prompt_user import _multiline_prompt
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.utilities.config import configure_if_first_time
from minisweagent.utils.serialize import UNSET, recursive_merge

'''
run/mini.py
    │
    ├── get_model()          创建 Model
    ├── get_environment()    创建 Environment
    └── get_agent()          创建 Agent
           │
           ▼
     agent.run(task)
           │
           ▼
     初始化 messages
           │
           ▼
     while True:
           │
           ├── step()
           │    │
           │    ├── query()
           │    │     └── model.query(messages)
           │    │           ├── litellm.completion()
           │    │           └── 解析工具调用
           │    │
           │    └── execute_actions()
           │          ├── env.execute(action)
           │          │     └── subprocess 执行 Bash
           │          └── 将结果加入 messages
           │
           ├── 保存执行轨迹
           │
           └── 判断是否退出
                ├── 否 → 下一轮
                └── 是 → 结束
'''

'''
agent.run(task)                  # 初始化任务、消息历史，启动 Agent
    │
    └── while True:             # 不断执行，直到满足退出条件
         │
         └── step()             # 执行一轮：模型决策 + 工具执行
              │
              ├── query()       # 检查预算，获取并记录模型回复
              │    │
              │    └── model.query(messages)
              │         │       # 将历史消息交给 Model，返回含 actions 的 message
              │         │
              │         ├── _prepare_messages_for_api()
              │         │       # 清理内部字段，转换成适合模型 API 的消息
              │         │
              │         ├── _query()
              │         │    │  # 组织模型请求参数
              │         │    └── litellm.completion()
              │         │           # 真正调用大模型，提供 Bash 工具定义
              │         │
              │         └── _parse_actions()
              │              │  # 从模型响应中提取工具调用
              │              └── parse_toolcall_actions()
              │                     # 校验并解析为 action，如 {"command": "pytest"}
              │
              └── execute_actions(message)
                   │            # 从 message 提取 actions，执行并记录反馈
                   │
                   ├── env.execute(action)
                   │    │       # 执行单个动作，返回 output、returncode 等
                   │    │
                   │    ├── _run()
                   │    │    │  # 管理 Shell 命令执行、超时及结果收集
                   │    │    └── subprocess.Popen()
                   │    │           # 创建操作系统子进程，执行 Bash 命令
                   │    │
                   │    └── _check_finished()
                   │            # 检查是否出现任务完成标记
                   │
                   └── model.format_observation_messages()
                        │       # 将执行结果转换成模型可接收的消息
                        │
                        └── format_toolcall_observation_messages()
                             │  # 按模板生成 tool 消息，关联工具调用 ID
                             │
                             ▼
                        add_messages()
                             │  # 将工具反馈追加到 self.messages
                             │
                             ▼
                  返回 run()，保存轨迹 save()
                             │
                             ▼
                  检查是否退出
                       │
                       ├── 否 → 下一轮 step()
                       │
                       └── 是 → 返回最终结果
                       
                       
                       


model.query()
    │
    └── 生成结束命令
         │
         ▼
env.execute(action)
    │
    ├── _run()                # 执行 echo 命令
    │
    └── _check_finished()     # 检查输出是否包含结束标记
         │
         └── raise Submitted  # 抛出任务完成信号
                │
                ▼
       DefaultAgent.run()
                │
                ├── except InterruptAgentFlow
                │       # 捕获信号，记录 exit 消息
                │
                ├── finally: save()
                │       # 保存执行轨迹
                │
                └── 返回最终结果
'''

DEFAULT_CONFIG_FILE = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT_FILE = global_config_dir / "last_mini_run.traj.json"


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c mini.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c mini.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml agent.mode=yolo[/bold green]
"""

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use",),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'litellm' or 'minisweagent.models.litellm_model.LitellmModel')", rich_help_panel="Advanced"),
    agent_class: str | None = typer.Option(None, "--agent-class", help="Agent class to use (e.g., 'interactive' or 'minisweagent.agents.interactive.InteractiveAgent')", rich_help_panel="Advanced"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class to use (e.g., 'local' or 'minisweagent.environments.local.LocalEnvironment')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
    output: Path | None = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
) -> Any:
    # fmt: on
    configure_if_first_time()

    # Build the config from the command line arguments
    console.print(f"Building agent config from specs: [bold green]{config_spec}[/bold green]")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "run": {
            "task": task or UNSET,
        },
        "agent": {
            "agent_class": agent_class or UNSET,
            "mode": "yolo" if yolo else UNSET,
            "cost_limit": cost_limit if cost_limit is not None else UNSET,
            "confirm_exit": False if exit_immediately else UNSET,
            "output_path": output or UNSET,
        },
        "model": {
            "model_class": model_class or UNSET,
            "model_name": model_name or UNSET,
        },
        "environment": {
            "environment_class": environment_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    # 「先检查配置中有没有用户指定的任务，如果没有，就在终端询问用户要做什么。」
    if (run_task := config.get("run", {}).get("task", UNSET)) is UNSET:
        console.print("[bold yellow]What do you want to do?")
        run_task = _multiline_prompt()
        console.print("[bold green]Got that, thanks![/bold green]")

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive")
    agent.run(run_task)
    if (output_path := config.get("agent", {}).get("output_path")):
        console.print(f"Saved trajectory to [bold green]'{output_path}'[/bold green]")
    return agent


if __name__ == "__main__":
    app()
