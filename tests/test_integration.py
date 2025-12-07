from __future__ import annotations

from llm.answer_generator import AnswerGenerator


class DummyLLMClient:
    def __init__(self, *args, **kwargs):
        self.prompt = ""

    def generate(self, prompt, system_prompt=None):
        self.prompt = prompt
        return "Mocked answer referencing [chunk_0::sub::0]"


def test_answer_generator_formats_response(monkeypatch, sample_config):
    monkeypatch.setattr("llm.answer_generator.LLMClient", lambda *args, **kwargs: DummyLLMClient())

    generator = AnswerGenerator(config_path=sample_config)
    local_results = [
        {
            "entity": "Dr. B. R. Ambedkar",
            "entity_score": 0.9,
            "chunks": [
                {
                    "chunk_id": "chunk_0::sub::0",
                    "score": 0.85,
                    "text": "Equality matters.",
                    "pages": [1],
                }
            ],
        }
    ]
    global_results = [
        {
            "community_id": 0,
            "points": [
                {"chunk_id": "chunk_0::sub::0", "score": 0.8, "text": "Community view on equality."}
            ],
        }
    ]

    response = generator.answer("What is equality?", local_results, global_results)
    assert "Mocked answer" in response["answer"]
    assert response["citations"] == ["chunk_0::sub::0"]

