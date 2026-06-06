"""BM25 retrieval over the citation-tagged corpus index."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from common import CORPUS_INDEX, read_jsonl

_TOKEN = re.compile(r"[a-z0-9§]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class Retriever:
    def __init__(self, snippets: list[dict] | None = None):
        self.snippets = snippets if snippets is not None else read_jsonl(CORPUS_INDEX)
        docs = [
            tokenize(f"{s['citation']} {s['title']} {s['text']}")
            for s in self.snippets
        ]
        self.bm25 = BM25Okapi(docs)

    def top(self, query: str, k: int = 8, guidance_floor: int = 2) -> list[dict]:
        """Top-k by BM25, guaranteeing at least `guidance_floor` guidance snippets.

        PACE Code provisions (GOWISELY, the caution, the arrest necessity test) are
        short and often out-ranked by long statute sections, yet are exactly the
        anchors we want models to cite. We reserve a couple of slots for the best
        guidance matches so they are always available to cite.
        """
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        chosen: list[int] = []
        guid = [i for i in order if self.snippets[i]["source_type"] == "guidance"]
        for i in guid[:guidance_floor]:
            chosen.append(i)
        for i in order:
            if len(chosen) >= k:
                break
            if i not in chosen:
                chosen.append(i)
        return [self.snippets[i] for i in chosen[:k]]
