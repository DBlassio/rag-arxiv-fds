# Our embedding module. Wraps SentenceTransformer to produce normalized vectors.


from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5" # D -> 384 dim.

_model: SentenceTransformer | None = None  # singleton: evita recargar el modelo en cada llamada


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# Our embedding function. Normalizes the vectors to unit length, which is important for cosine similarity.
def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    return model.encode(
        texts,
        normalize_embeddings=True,  
        batch_size=32,
        show_progress_bar=False)


if __name__ == "__main__":
    vecs = embed_texts(["retrieval augmented generation", "transformer attention"])
    print(vecs.shape)   # esperado: (2, 384)