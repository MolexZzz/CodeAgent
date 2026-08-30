from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from memcodeagent.agent import AgentConfig, CodingAgent

app = typer.Typer(help="MemCodeAgent: a lightweight CLI coding agent.")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Workspace root for the default chat."),
    max_steps: int = typer.Option(8, "--max-steps", help="Maximum steps per continuation budget."),
    approve: bool = typer.Option(True, "--approve/--no-approve", help="Ask before file edits and commands."),
    protect_tests: bool = typer.Option(True, "--protect-tests/--no-protect-tests", help="Protect tests that existed before the task started."),
) -> None:
    """Enter the interactive agent when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        config = AgentConfig(
            workspace=workspace.resolve(),
            max_steps=max_steps,
            approval_required=approve,
            protect_existing_tests=protect_tests,
        )
        agent = CodingAgent(config=config, console=console)
        agent.chat()


@app.command()
def run(
    task: str = typer.Argument(..., help="Programming task for the agent."),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Workspace root."),
    max_steps: int = typer.Option(8, "--max-steps", help="Maximum agent loop steps."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan without writing files or running commands."),
) -> None:
    """Run the coding agent on a task."""
    config = AgentConfig(workspace=workspace.resolve(), max_steps=max_steps, dry_run=dry_run)
    agent = CodingAgent(config=config, console=console)
    result = agent.run(task)
    console.rule("[bold green]Final")
    console.print(Markdown(result))

@app.command()
def chat(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Workspace root."),
    max_steps: int = typer.Option(8, "--max-steps", help="Deprecated: use --max-error-retries instead."),
    max_error_retries: int = typer.Option(10, "--max-error-retries", help="Maximum retry attempts per error."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan without writing files or running commands."),
    approve: bool = typer.Option(True, "--approve/--no-approve", help="Ask before each tool call."),
    protect_tests: bool = typer.Option(True, "--protect-tests/--no-protect-tests", help="Protect tests that existed before the task started."),
) -> None:
    """Start an interactive chat session with the coding agent."""
    config = AgentConfig(
        workspace=workspace.resolve(),
        max_steps=max_steps,
        max_error_retries=max_error_retries,
        dry_run=dry_run,
        approval_required=approve,
        protect_existing_tests=protect_tests,
    )
    agent = CodingAgent(config=config, console=console)
    agent.chat()


@app.command()
def index(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Workspace root."),
) -> None:
    """Build or refresh the project memory index."""
    config = AgentConfig(workspace=workspace.resolve())
    agent = CodingAgent(config=config, console=console)
    summary = agent.index_workspace()
    console.print(summary)

"""
main()：如果你没写子命令，就直接进交互式聊天模式
run task：一次性执行一个编程任务，跑完后把最终结果打印出来
chat：进入 REPL 交互模式，支持持续对话
index：重建/刷新项目的记忆索引

1.从命令行拿参数，比如 --workspace、--max-steps、--dry-run
2.组装成 AgentConfig
3.创建 CodingAgent(config=config, console=console)
4.调用对应方法：- agent.chat()
- agent.run(task)
- agent.index_workspace()
"""

if __name__ == "__main__":
    app()
