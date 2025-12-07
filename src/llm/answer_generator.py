from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from llm.llm_client import LLMClient, LLMConfig
from llm.prompt_templates import ANSWER_SYSTEM_PROMPT, build_answer_prompt


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class AnswerGenerator:
    def __init__(self, config_path: Path = Path("config.yaml")) -> None:
        cfg = load_config(config_path)
        llm_cfg = cfg.get("llm", {})
        self.client = LLMClient(
            LLMConfig(
                model=llm_cfg.get("model", "llama3"),
                temperature=llm_cfg.get("temperature", 0.2),
                max_tokens=llm_cfg.get("max_tokens", 512),
            )
        )

    @staticmethod
    def _collect_citations(local_results: List[Dict], global_results: List[Dict]) -> List[str]:
        citations = set()
        for item in local_results:
            for chunk in item.get("chunks", []):
                citations.add(chunk["chunk_id"])
        for community in global_results:
            for point in community.get("points", []):
                citations.add(point["chunk_id"])
        return sorted(citations)

    def answer(
        self,
        query: str,
        local_results: List[Dict],
        global_results: List[Dict],
    ) -> Dict[str, Any]:
        prompt = build_answer_prompt(query, local_results, global_results)
        response = self.client.generate(prompt, system_prompt=ANSWER_SYSTEM_PROMPT)
        return {
            "answer": response,
            "citations": self._collect_citations(local_results, global_results),
        }
