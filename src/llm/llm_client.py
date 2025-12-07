from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import ollama


@dataclass
class LLMConfig:
    model: str = "llama3"
    temperature: float = 0.2
    max_tokens: int = 512


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.config.model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        return response["message"]["content"].strip()


def run_llm(prompt: str, system_prompt: Optional[str] = None, **kwargs: Dict[str, Any]) -> str:
    client = LLMClient()
    return client.generate(prompt, system_prompt=system_prompt)
