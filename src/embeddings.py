# src/embeddings.py
# ---------------------------------------------------------
# Handles loading the embedding model.
# Single responsibility: provide a ready-to-use embedding
# model that converts text → vectors.
# ---------------------------------------------------------

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME


def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads and returns the HuggingFace embedding model.

    The model downloads automatically on first run (~90MB).
    After first download, it loads from local cache instantly.

    Returns:
        HuggingFaceEmbeddings: ready-to-use embedding model
    """

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(First run downloads ~90MB — subsequent runs are instant)")

    # HuggingFaceEmbeddings is a LangChain wrapper that:
    # 1. Downloads the model from HuggingFace Hub (first time only)
    # 2. Loads it into memory
    # 3. Provides an .embed_documents() and .embed_query() method
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,

        # model_kwargs: passed directly to the underlying model
        # device: "cpu" means run on CPU, not GPU
        # For most laptops, CPU is the right choice
        model_kwargs={"device": "cpu"},

        # encode_kwargs: controls the encoding process
        # normalize_embeddings: True makes cosine similarity
        # work correctly by ensuring all vectors have length 1
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Embedding model loaded successfully.")
    return embedding_model