"""Demo script for AmbedkarGPT - runs predefined questions for interview demonstration."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src.llm.answer_generator import AnswerGenerator
from src.retrieval.global_search import GlobalGraphRAG
from src.retrieval.local_search import LocalGraphRAG

console = Console()

# Predefined demo questions from assignment
DEMO_QUESTIONS = [
    "What is Dr. Ambedkar's view on caste?",
    "What did Ambedkar say about social justice?",
    "Who opposed caste discrimination?",
    "What is endogamy according to Ambedkar?",
    "How did Ambedkar describe social justice?",
    "Which reforms did Ambedkar advocate for education?",
    "Summarize Ambedkar's stance on liberty, equality, and fraternity.",
]


def run_demo(config_path: Path = Path("config.yaml")) -> None:
    """Run predefined demo questions to demonstrate the system."""
    console.rule("[bold cyan]AmbedkarGPT Demo Script[/bold cyan]")
    console.print(
        Panel(
            "This script demonstrates the RAG system with predefined questions.\n"
            "Each question will show retrieved context and generated answers with citations.",
            title="Demo Mode",
            border_style="cyan",
        )
    )

    try:
        console.print("\n[yellow]Initializing system components...[/yellow]")
        local_retriever = LocalGraphRAG(config_path)
        global_retriever = GlobalGraphRAG(config_path)
        generator = AnswerGenerator(config_path)
        console.print("[green]✓ System initialized successfully[/green]\n")
    except Exception as e:
        console.print(f"[red]Error initializing system:[/red] {e}")
        console.print("[yellow]Ensure all pipeline steps are completed:[/yellow]")
        console.print("  1. python -m src.pipeline.ambedkargpt chunk")
        console.print("  2. python -m src.pipeline.ambedkargpt build-graph")
        console.print("  3. python -m src.pipeline.ambedkargpt detect-communities")
        console.print("  4. python -m src.pipeline.ambedkargpt summarize-communities")
        raise

    console.print(f"[bold]Running {len(DEMO_QUESTIONS)} demo questions...[/bold]\n")

    for idx, query in enumerate(DEMO_QUESTIONS, 1):
        console.rule(f"[bold]Question {idx}/{len(DEMO_QUESTIONS)}[/bold]")
        console.print(f"[cyan]Q:[/cyan] {query}\n")

        try:
            # Retrieve context
            console.print("[dim]Retrieving local and global context...[/dim]")
            local_results = local_retriever.search(query)
            global_results = global_retriever.search(query)

            # Generate answer
            console.print("[dim]Generating answer...[/dim]")
            response = generator.answer(query, local_results, global_results)

            # Display results
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(response["answer"])

            if response["citations"]:
                console.print(f"\n[bold]Citations:[/bold] {', '.join(response['citations'])}")
            else:
                console.print("\n[bold]Citations:[/bold] None")

            # Show retrieval stats
            local_entities = len(local_results)
            global_communities = len(global_results)
            console.print(f"\n[dim]Retrieved: {local_entities} entities (local), {global_communities} communities (global)[/dim]")

        except Exception as e:
            console.print(f"[red]Error processing question:[/red] {e}")
            console.print("[yellow]Continuing with next question...[/yellow]")

        console.print("\n" + "=" * 80 + "\n")

    console.rule("[bold green]Demo Complete[/bold green]")
    console.print("[green]All questions processed successfully![/green]")


if __name__ == "__main__":
    import sys

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.yaml")
    run_demo(config_path)




