# src/vector_store.py
# ---------------------------------------------------------
# Handles building, saving, loading, and searching
# the FAISS vector store.
# Single responsibility: store chunks as vectors and
# retrieve the most relevant ones for a given question.
# ---------------------------------------------------------

import os
from langchain_community.vectorstores import FAISS
from config import FAISS_INDEX_PATH, TOP_K_RESULTS


def build_vector_store(chunks: list,
                       embedding_model) -> FAISS:
    """
    Converts text chunks into vectors and stores them
    in a FAISS index. Saves the index to disk.

    Args:
        chunks: list of Document objects from text_chunker
        embedding_model: loaded HuggingFaceEmbeddings model

    Returns:
        FAISS: the built and saved vector store
    """

    print(f"Building vector store from {len(chunks)} chunks...")

    # FAISS.from_documents() does THREE things in one call:
    # 1. Extracts .page_content from each Document
    # 2. Passes each text through the embedding model → vectors
    # 3. Stores all (vector, text, metadata) triples in FAISS index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    # Save the FAISS index to disk so we don't rebuild
    # every time the user asks a question
    # Creates two files:
    #   faiss_index/index.faiss  ← the actual vectors
    #   faiss_index/index.pkl    ← chunk texts + metadata
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_PATH)

    print(f"Vector store saved to '{FAISS_INDEX_PATH}/'")
    return vector_store


def load_vector_store(embedding_model) -> FAISS:
    """
    Loads a previously saved FAISS index from disk.
    Use this instead of rebuilding when a PDF is
    already processed.

    Args:
        embedding_model: same model used to build the store

    Returns:
        FAISS: the loaded vector store, ready to search
    """

    # Check if a saved index actually exists
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"No FAISS index found at '{FAISS_INDEX_PATH}'. "
            "Please upload and process a PDF first."
        )

    # allow_dangerous_deserialization=True is required by
    # newer LangChain versions — it acknowledges you trust
    # the source of the saved index file (which you do,
    # since YOU saved it)
    vector_store = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    print("Vector store loaded from disk.")
    return vector_store


def search_similar_chunks(vector_store: FAISS,
                          question: str) -> list:
    """
    Searches the vector store for chunks most relevant
    to the user's question.

    Args:
        vector_store: loaded FAISS index
        question: the user's plain English question

    Returns:
        list of Document objects — the top K most relevant chunks
        Each Document has .page_content and .metadata
    """

    # similarity_search() does TWO things:
    # 1. Embeds the question text → question vector
    # 2. Finds top K chunks whose vectors are closest
    #    to the question vector (cosine similarity)
    relevant_chunks = vector_store.similarity_search(
        query=question,
        k=TOP_K_RESULTS      # Return top 3 most relevant chunks
    )

    return relevant_chunks


def vector_store_exists() -> bool:
    """
    Quick check: has a vector store been built yet?
    Used by app.py to decide whether to show the chat input.

    Returns:
        bool: True if index files exist on disk
    """

    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    return os.path.exists(index_file)