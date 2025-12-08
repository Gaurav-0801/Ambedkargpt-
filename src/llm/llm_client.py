from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import ollama
from rich.console import Console

console = Console()


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

        try:
            response = ollama.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )
            if "message" not in response or "content" not in response["message"]:
                raise ValueError(f"Unexpected response format from Ollama: {response}")
            return response["message"]["content"].strip()
        except ollama.ResponseError as e:
            console.print(f"[red]Ollama API error:[/red] {e}")
            raise RuntimeError(f"Failed to generate response from Ollama model '{self.config.model}'. Ensure Ollama is running and the model is available.") from e
        except ConnectionError as e:
            console.print(f"[red]Connection error:[/red] {e}")
            raise RuntimeError("Failed to connect to Ollama. Ensure Ollama is running (e.g., 'ollama serve').") from e
        except Exception as e:
            console.print(f"[red]Unexpected error during LLM generation:[/red] {e}")
            raise


def run_llm(prompt: str, system_prompt: Optional[str] = None, **kwargs: Dict[str, Any]) -> str:
    client = LLMClient()
    return client.generate(prompt, system_prompt=system_prompt)
