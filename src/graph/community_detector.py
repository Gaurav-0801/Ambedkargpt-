from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import community as community_louvain
import yaml
from rich.console import Console
import typer

console = Console()
app = typer.Typer(no_args_is_help=True)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_graph(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def persist_graph(graph, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(graph, fh)


@app.command()
def detect(config: Path = typer.Option(Path("config.yaml"), "--config", "-c")) -> None:
    cfg = load_config(config)
    paths = cfg["paths"]

    graph_path = Path(paths["graph"])
    partition_path = Path(paths["community_partition"])
    console.rule("[bold]Community Detection[/bold]")
    graph = load_graph(graph_path)
    console.log(f"Loaded graph with {graph.number_of_nodes()} nodes")

    partition = community_louvain.best_partition(graph)
    console.log(f"Detected {len(set(partition.values()))} communities")

    community_groups = defaultdict(list)
    for node, community_id in partition.items():
        graph.nodes[node]["community"] = community_id
        community_groups[community_id].append(node)

    persist_graph(graph, graph_path)
    console.log(f"Updated graph saved to {graph_path}")

    partition_path.parent.mkdir(parents=True, exist_ok=True)
    with partition_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"partition": partition, "communities": community_groups},
            fh,
            indent=2,
        )
    console.log(f"Community partition written to {partition_path}")


if __name__ == "__main__":
    app()
