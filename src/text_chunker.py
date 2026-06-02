# src/text_chunker.py
# ---------------------------------------------------------
# Handles splitting extracted PDF text into overlapping
# chunks that can be stored and searched efficiently.
# Single responsibility: take page list, return chunk list.
# ---------------------------------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(extracted_pages: list) -> list:
    """
    Takes the list of page dictionaries from pdf_processor
    and splits each page's text into smaller overlapping chunks.

    Args:
        extracted_pages: list of dicts with keys:
                         page_num, text, char_count

    Returns:
        list of LangChain Document objects, each with:
            - page_content: the chunk text
            - metadata: dict containing source page number
    """

    # Initialize the text splitter with our config settings
    # separators: ordered list of what to split on — tries
    # each one in order until chunk fits within chunk_size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # Max chars per chunk
        chunk_overlap=CHUNK_OVERLAP,    # Overlap between chunks
        separators=[
            "\n\n",   # First try: split at paragraph breaks
            "\n",     # Second try: split at line breaks
            ". ",     # Third try: split at sentence endings
            " ",      # Fourth try: split at word boundaries
            ""        # Last resort: split at any character
        ],
        length_function=len,    # Use len() to measure chunk size
    )

    # This list will hold all Document objects across all pages
    all_chunks = []

    for page in extracted_pages:

        # Create chunks from this page's text
        # create_documents() returns a list of Document objects
        # Each Document has .page_content and .metadata
        chunks = text_splitter.create_documents(
            texts=[page["text"]],   # Text to split — must be a list
            metadatas=[{            # Metadata stored WITH each chunk
                "page_num": page["page_num"],
                "source": f"Page {page['page_num']}"
            }]
        )

        # Add this page's chunks to our master list
        all_chunks.extend(chunks)

    return all_chunks


def get_chunks_summary(chunks: list) -> dict:
    """
    Returns a summary of the chunking results.
    Useful for debugging and for showing the user what happened.

    Args:
        chunks: list of Document objects from create_chunks()

    Returns:
        dict with total_chunks, avg_chunk_size, min_size, max_size
    """

    # Extract just the lengths of each chunk's text
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]

    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(chunk_sizes) // len(chunk_sizes),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes)
    }


def preview_chunks(chunks: list, num_preview: int = 3) -> None:
    """
    Prints a preview of the first N chunks.
    Use this during development to verify chunking is working.

    Args:
        chunks: list of Document objects
        num_preview: how many chunks to preview (default 3)
    """

    print(f"\n{'='*50}")
    print(f"CHUNK PREVIEW — showing {num_preview} of {len(chunks)}")
    print(f"{'='*50}")

    for i, chunk in enumerate(chunks[:num_preview]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Size: {len(chunk.page_content)} chars")
        print(f"Content preview: {chunk.page_content[:150]}...")
        print()