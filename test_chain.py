# test_chain.py

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pdf_processor import extract_text_from_pdf
from src.text_chunker import create_chunks
from src.embeddings import load_embedding_model
from src.vector_store import build_vector_store, search_similar_chunks
from src.qa_chain import answer_question

print("=" * 50)
print("TESTING HR Q&A BOT BACKEND")
print("=" * 50)

# ↓↓↓ CHANGE THIS TO YOUR ACTUAL PDF FILENAME ↓↓↓
PDF_PATH = "uploaded_pdfs/hr_policy.pdf"
# ↑↑↑ CHANGE THIS TO YOUR ACTUAL PDF FILENAME ↑↑↑

# Step 1 — Extract
print("\n[1/5] Extracting text from PDF...")
pages = extract_text_from_pdf(PDF_PATH)
print(f"      ✅ Extracted {len(pages)} pages")

# Step 2 — Chunk
print("\n[2/5] Creating chunks...")
chunks = create_chunks(pages)
print(f"      ✅ Created {len(chunks)} chunks")

# Step 3 — Load model + build vector store
print("\n[3/5] Loading embedding model...")
model = load_embedding_model()
print("\n[4/5] Building vector store...")
store = build_vector_store(chunks, model)
print(f"      ✅ Vector store built and saved")

# Step 4 — Search
print("\n[5/5] Testing a question...")
question = "What is the leave policy?"
results = search_similar_chunks(store, question)
print(f"      ✅ Found {len(results)} relevant chunks")

# Step 5 — Answer
print("\n" + "=" * 50)
print("QUESTION:", question)
print("=" * 50)
output = answer_question(question, results)
print("\nANSWER:")
print(output["answer"])
print(f"\nSource pages: {output['source_pages']}")
print("\n✅ Backend test complete!")