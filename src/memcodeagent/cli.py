from pathlib import Path

import typer
from rich.console import Console

from memcodeagent.agent import AgentConfig, CodingAgent

app = typer.Typer(help="MemCodeAgent: a lightweight CLI coding agent.")
console = Console()


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
    console.print(result)


@app.command()
def chat(
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Workspace root."),
    max_steps: int = typer.Option(8, "--max-steps", help="Maximum agent loop steps per turn."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan without writing files or running commands."),
) -> None:
    """Start an interactive chat session with the coding agent."""
    config = AgentConfig(workspace=workspace.resolve(), max_steps=max_steps, dry_run=dry_run)
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


if __name__ == "__main__":
    app()
