# Our embedding module.
# Wraps SentenceTransformer to produce normalized vectors.

from sentence_transformers import SentenceTransformer
import numpy as np


# Local model baked into the Docker image
MODEL_PATH = "/opt/models/bge-small-en-v1.5"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model

    if _model is None:
        _model = SentenceTransformer(
            MODEL_PATH,
            local_files_only=True,
        )

    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()

    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )


if __name__ == "__main__":
    vecs = embed_texts([
        "retrieval augmented generation",
        "transformer attention",
    ])

    print(vecs.shape)  # esperado: (2, 384)