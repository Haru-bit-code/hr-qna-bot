# config.py
# ---------------------------------------------------------
# Central configuration — updated to use Groq API
# ---------------------------------------------------------

import os
from dotenv import load_dotenv

load_dotenv()

# ----- API KEYS -----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----- PDF PROCESSING -----
UPLOAD_FOLDER = "uploaded_pdfs"

# ----- TEXT CHUNKING SETTINGS -----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ----- VECTOR SEARCH SETTINGS -----
TOP_K_RESULTS = 3
FAISS_INDEX_PATH = "faiss_index"

# ----- EMBEDDING MODEL -----
# Runs locally — completely free, no API needed
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ----- LLM SETTINGS (GROQ) -----
# Groq supports multiple models — llama is fast and accurate
# Options:
#   "llama-3.1-8b-instant"    ← fastest
#   "llama-3.1-70b-versatile" ← most accurate
#   "mixtral-8x7b-32768"      ← large context window
LLM_MODEL_NAME = "llama-3.1-8b-instant"

# Max tokens in the LLM response
MAX_TOKENS = 1024

# ----- UI SETTINGS -----
APP_TITLE = "HR Policy Q&A Bot"
APP_DESCRIPTION = (
    "Upload your company's HR policy PDF and ask "
    "questions in plain English."
)