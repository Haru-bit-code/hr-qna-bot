# src/pdf_processor.py
# ---------------------------------------------------------
# Handles all PDF reading and text extraction.
# Single responsibility: take a PDF path, return page text.
# ---------------------------------------------------------

import fitz          # PyMuPDF — for reading PDFs
import os            # For file path operations
from config import UPLOAD_FOLDER   # Our central config


def save_uploaded_pdf(uploaded_file) -> str:
    """
    Saves a Streamlit uploaded file object to disk.

    Args:
        uploaded_file: the file object from st.file_uploader()

    Returns:
        str: the full file path where the PDF was saved
    """

    # Make sure the upload folder exists
    # exist_ok=True means: don't crash if folder already exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Build the full save path
    # Example: "uploaded_pdfs/hr_policy.pdf"
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    # Open the path in write-binary mode ("wb")
    # PDFs are binary files — not plain text — so we use "wb"
    # uploaded_file.getbuffer() reads the raw bytes from Streamlit
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Opens a PDF and extracts text from every page.

    Args:
        pdf_path: full path to the PDF file on disk

    Returns:
        list of dicts, each containing:
            - page_num (int): human-friendly page number
            - text (str): full text content of that page
            - char_count (int): number of characters extracted
    """

    # Validate that the file actually exists before opening
    # Prevents confusing errors later in the pipeline
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at path: {pdf_path}"
        )

    extracted_pages = []

    # Open the PDF — fitz.open() parses the binary format
    # and builds a Document object representing the whole file
    doc = fitz.open(pdf_path)

    # Check if the document is empty
    if len(doc) == 0:
        raise ValueError("The PDF file has no pages.")

    for page_num in range(len(doc)):

        # Load one page into memory
        page = doc.load_page(page_num)

        # Extract all text from this page
        # PyMuPDF reads content streams + font maps internally
        text = page.get_text()

        # Clean up whitespace
        text = text.strip()

        # Skip blank pages — they add noise to our system
        if not text:
            continue

        extracted_pages.append({
            "page_num": page_num + 1,   # 1-indexed for humans
            "text": text,
            "char_count": len(text)
        })

    # Always close the document to free memory
    doc.close()

    # Safety check — warn if almost nothing was extracted
    if len(extracted_pages) == 0:
        raise ValueError(
            "No text could be extracted. "
            "This may be a scanned (image-based) PDF. "
            "Please use a digital PDF."
        )

    return extracted_pages


def get_pdf_summary(extracted_pages: list) -> dict:
    """
    Returns a quick summary of what was extracted.
    Useful for showing the user a confirmation message.

    Args:
        extracted_pages: the list returned by extract_text_from_pdf()

    Returns:
        dict with total_pages, total_characters, avg_chars_per_page
    """

    total_chars = sum(p["char_count"] for p in extracted_pages)

    return {
        "total_pages": len(extracted_pages),
        "total_characters": total_chars,
        "avg_chars_per_page": total_chars // len(extracted_pages)
    }