from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import typer
import yaml
from rich.console import Console

from llm.llm_client import run_llm

console = Console()
app = typer.Typer(no_args_is_help=True)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_graph(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_chunks(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def format_prompt(community_id: int, payload: Dict[str, Any]) -> str:
    entities_section = "\n".join(payload["entities"])
    relations_section = "\n".join(payload["relations"]) or "None captured."
    evidence_section = "\n".join(payload["evidence"]) or "No direct citations."

    return f"""
You are helping to summarize a knowledge-graph community derived from Dr. B. R. Ambedkar's works.

Community ID: {community_id}

Entities:
{entities_section}

Relations:
{relations_section}

Context Snippets:
{evidence_section}

Provide a concise summary (3-5 bullet points) highlighting the key themes, relationships, and insights from this community. Mention important entities by name and reference any notable relationships or events.
"""


def build_community_payload(graph, community_id: int, chunk_lookup: Dict[str, str]) -> Dict[str, Any]:
    entities: List[str] = []
    relations: List[str] = []
    evidence: List[str] = []

    for node, data in graph.nodes(data=True):
        if data.get("community") != community_id:
            continue
        snippet = "; ".join(data.get("examples", []))[:240]
        entities.append(
            f"- {node} ({data.get('label', 'N/A')}) | mentions: {data.get('count', 0)} | pages: {data.get('pages', [])} | context: {snippet}"
        )

    for src, tgt, data in graph.edges(data=True):
        src_comm = graph.nodes[src].get("community")
        tgt_comm = graph.nodes[tgt].get("community")
        if src_comm != community_id or tgt_comm != community_id:
            continue
        relations.append(f"- {src} --[{', '.join(data.get('relations', []))}]--> {tgt}")
        for item in data.get("evidence", [])[:3]:
            chunk_id = item.get("chunk_id")
            chunk_text = chunk_lookup.get(chunk_id, "")[:320]
            evidence.append(f"[{chunk_id}] {item.get('text', '')} | Chunk: {chunk_text}")

    return {"entities": entities, "relations": relations, "evidence": evidence}


@app.command()
def summarize(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    cfg = load_config(config)
    paths = cfg["paths"]

    graph = load_graph(Path(paths["graph"]))
    chunk_data = load_chunks(Path(paths["chunks"]))
    chunk_lookup = {c["id"]: c["text"] for c in chunk_data["chunks"]}

    reports_path = Path(paths["community_reports"])
    reports: List[Dict[str, Any]] = []

    communities = {data.get("community") for _, data in graph.nodes(data=True) if data.get("community") is not None}
    console.rule("[bold]Community Summaries[/bold]")
    for community_id in sorted(communities):
        payload = build_community_payload(graph, community_id, chunk_lookup)
        prompt = format_prompt(community_id, payload)
        summary = run_llm(prompt)
        reports.append(
            {
                "community_id": community_id,
                "summary": summary.strip(),
                "entities": payload["entities"],
                "relations": payload["relations"],
            }
        )
        console.log(f"Generated summary for community {community_id}")

    reports_path.parent.mkdir(parents=True, exist_ok=True)
    with reports_path.open("w", encoding="utf-8") as fh:
        json.dump({"reports": reports}, fh, indent=2)
    console.log(f"Community summaries stored at {reports_path}")


if __name__ == "__main__":
    app()
