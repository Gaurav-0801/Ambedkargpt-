from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt

from src.chunking import semantic_chunker
from src.graph import community_detector, graph_builder, summarizer
from src.llm.answer_generator import AnswerGenerator
from src.retrieval.global_search import GlobalGraphRAG
from src.retrieval.local_search import LocalGraphRAG

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def chunk(config: Path = typer.Option(Path("config.yaml"), "--config", "-c")) -> None:
    semantic_chunker.run(config)


@app.command("build-graph")
def build_graph(config: Path = typer.Option(Path("config.yaml"), "--config", "-c")) -> None:
    graph_builder.main(config)


@app.command("detect-communities")
def detect_communities(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c")
) -> None:
    community_detector.detect(config)


@app.command("summarize-communities")
def summarize_communities(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c")
) -> None:
    summarizer.summarize(config)


@app.command()
def run(config: Path = typer.Option(Path("config.yaml"), "--config", "-c")) -> None:
    console.rule("[bold cyan]AmbedkarGPT Interactive Mode[/bold cyan]")
    try:
        local_retriever = LocalGraphRAG(config)
        global_retriever = GlobalGraphRAG(config)
        generator = AnswerGenerator(config)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error initializing system:[/red] {e}")
        console.print("[yellow]Ensure all pipeline steps are completed:[/yellow]")
        console.print("  1. python -m src.pipeline.ambedkargpt chunk")
        console.print("  2. python -m src.pipeline.ambedkargpt build-graph")
        console.print("  3. python -m src.pipeline.ambedkargpt detect-communities")
        console.print("  4. python -m src.pipeline.ambedkargpt summarize-communities")
        raise

    while True:
        try:
            query = Prompt.ask("\nAsk AmbedkarGPT (type 'exit' to quit)")
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower().strip() in {"exit", "quit"}:
            console.print("Session ended.")
            break

        try:
            local_results = local_retriever.search(query)
            global_results = global_retriever.search(query)
            response = generator.answer(query, local_results, global_results)

            console.print("\n[bold]Answer[/bold]")
            console.print(response["answer"])
            if response["citations"]:
                console.print(f"[bold]Citations:[/bold] {', '.join(response['citations'])}")
            else:
                console.print("[bold]Citations:[/bold] None")
        except Exception as e:
            console.print(f"[red]Error processing query:[/red] {e}")
            console.print("[yellow]Please try again or check system logs.[/yellow]")


if __name__ == "__main__":
    app()
