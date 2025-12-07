from __future__ import annotations

from typing import Dict, List


ANSWER_SYSTEM_PROMPT = (
    "You are AmbedkarGPT, an expert assistant grounded in Dr. B. R. Ambedkar's works. "
    "Answer concisely, rely only on the provided context, and cite chunk ids like [chunk_1::sub::0]."
)


def _format_local_context(local_results: List[Dict]) -> str:
    if not local_results:
        return "No local entity context retrieved."

    sections = []
    for item in local_results:
        header = f"Entity: {item['entity']} (score={item['entity_score']:.3f})"
        chunk_lines = []
        for chunk in item["chunks"]:
            snippet = chunk["text"][:400].replace("\n", " ")
            chunk_lines.append(
                f"[{chunk['chunk_id']}] score={chunk['score']:.3f} pages={chunk.get('pages', [])}: {snippet}"
            )
        sections.append(f"{header}\n" + "\n".join(chunk_lines))
    return "\n\n".join(sections)


def _format_global_context(global_results: List[Dict]) -> str:
    if not global_results:
        return "No community summaries retrieved."

    sections = []
    for community in global_results:
        lines = []
        for point in community["points"]:
            snippet = point["text"][:400].replace("\n", " ")
            lines.append(
                f"[{point['chunk_id']}] score={point['score']:.3f} community={community['community_id']}: {snippet}"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_answer_prompt(
    query: str,
    local_results: List[Dict],
    global_results: List[Dict],
) -> str:
    local_context = _format_local_context(local_results)
    global_context = _format_global_context(global_results)

    return f"""
Question: {query}

Local Graph Context:
{local_context}

Global Community Context:
{global_context}

Instructions:
- Use both local and global context to answer.
- Prefer directly cited facts; avoid speculation.
- Provide 1-2 short paragraphs.
- Cite supporting chunk ids inline using [chunk_id].
"""

