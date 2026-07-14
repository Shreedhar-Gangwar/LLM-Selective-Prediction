"""TF-IDF retrieval of few-shot examples.

A fixed prompt shows the model the same examples for every message. Retrieval instead
picks the shots that are *nearest to the message being classified*, so the prompt
demonstrates exactly the confusable intents in play (e.g. a delivery complaint retrieves
'card arrival' / 'card delivery estimate' neighbours, which is precisely the pair the
model was getting wrong).

Character n-gram TF-IDF is deliberate: banking77 messages are short and full of near-
paraphrases, and character n-grams are robust to the typos and inflection that word
tokens miss. No embedding model is needed, which keeps VRAM for the LLM.

Split hygiene: the retrieval index is built from the few-shot pool ONLY, so a calibration
or test message can never retrieve itself or leak across splits.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.data import Example, load_split


class ShotRetriever:
    """Nearest-neighbour few-shot selector over the few-shot pool."""

    def __init__(self, pool: list[Example] | None = None):
        self.pool = pool if pool is not None else load_split("fewshot_pool")
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2
        )
        self.matrix = self.vectorizer.fit_transform([e.text for e in self.pool])

    def retrieve(self, text: str, k: int) -> list[Example]:
        """The k most similar pool examples, ordered *least* similar first.

        Ascending order puts the closest example immediately before the message being
        classified — the position an autoregressive model attends to most strongly.
        """
        sims = linear_kernel(self.vectorizer.transform([text]), self.matrix).ravel()
        top = sims.argsort()[-k:]  # ascending similarity
        return [self.pool[i] for i in top]
