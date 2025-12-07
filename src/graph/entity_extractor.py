from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import spacy


@lru_cache(maxsize=1)
def _load_model():
    return spacy.load("en_core_web_sm")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def analyze_text(
    text: str,
    allowed_types: Optional[List[str]] = None,
    min_length: int = 3,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Extract entities and lightweight relations from the provided text."""

    nlp = _load_model()
    doc = nlp(text)

    entities: List[Dict[str, str]] = []
    for ent in doc.ents:
        if allowed_types and ent.label_ not in allowed_types:
            continue
        normalized = _normalize(ent.text)
        if len(normalized) < min_length:
            continue
        entities.append({"text": normalized, "label": ent.label_})

    relations: List[Dict[str, str]] = []
    for sent in doc.sents:
        sent_entities = [
            ent
            for ent in sent.ents
            if (not allowed_types or ent.label_ in allowed_types)
        ]
        if len(sent_entities) < 2:
            continue

        relation_label = sent.root.lemma_.lower() if sent.root else "related_to"
        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                src = _normalize(sent_entities[i].text)
                tgt = _normalize(sent_entities[j].text)
                if len(src) < min_length or len(tgt) < min_length:
                    continue
                relations.append(
                    {
                        "source": src,
                        "target": tgt,
                        "relation": relation_label or "related_to",
                        "sentence": sent.text.strip(),
                    }
                )

    return entities, relations
