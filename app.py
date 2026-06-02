# app.py
# ---------------------------------------------------------
# HR Policy Q&A Bot — Main Streamlit Application
# Run with: streamlit run app.py
# ---------------------------------------------------------

import streamlit as st
import os
import sys

# Make sure Python can find our src/ modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pdf_processor import (
    save_uploaded_pdf,
    extract_text_from_pdf,
    get_pdf_summary
)
from src.text_chunker import create_chunks, get_chunks_summary
from src.embeddings import load_embedding_model
from src.vector_store import (
    build_vector_store,
    load_vector_store,
    search_similar_chunks,
    vector_store_exists
)
from src.qa_chain import answer_question
from config import APP_TITLE, APP_DESCRIPTION

# ---------------------------------------------------------
# PAGE CONFIG — must be the VERY FIRST streamlit command
# ---------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
    <style>
    .answer-box {
        background-color: #f0f7ff;
        border-left: 4px solid #2196F3;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #1a1a2e;
    }
    .source-badge {
        background-color: #e8f5e9;
        border: 1px solid #4CAF50;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #2e7d32;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #e65100;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------
st.title("🤖 " + APP_TITLE)
st.markdown(f"*{APP_DESCRIPTION}*")
st.divider()

# ---------------------------------------------------------
# LOAD EMBEDDING MODEL ONCE — cached across all reruns
# ---------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    """
    Loads embedding model once and caches it.
    Without this, the model reloads on every interaction.
    """
    with st.spinner("Loading AI model... (first time only)"):
        return load_embedding_model()

embedding_model = get_embedding_model()

# ---------------------------------------------------------
# SIDEBAR — PDF UPLOAD AND PROCESSING
# ---------------------------------------------------------
with st.sidebar:
    st.header("📄 Document Setup")
    st.markdown("Upload your HR policy PDF to get started.")

    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],
        help="Upload your company's HR policy document"
    )

    if uploaded_file is not None:

        # Check if this exact file was already processed
        already_processed = (
            st.session_state.get("processed_filename") ==
            uploaded_file.name
        )

        if not already_processed:

            if st.button("⚙️ Process PDF", type="primary"):

                try:
                    progress = st.progress(0)
                    status = st.empty()

                    # Step 1 — Save
                    status.text("Saving PDF...")
                    pdf_path = save_uploaded_pdf(uploaded_file)
                    progress.progress(25)

                    # Step 2 — Extract
                    status.text("Extracting text...")
                    pages = extract_text_from_pdf(pdf_path)
                    progress.progress(50)

                    # Step 3 — Chunk
                    status.text("Creating chunks...")
                    chunks = create_chunks(pages)
                    progress.progress(65)

                    # Step 4 — Embed and store
                    status.text("Building vector store...")
                    build_vector_store(chunks, embedding_model)
                    progress.progress(100)

                    # Clear progress UI
                    status.empty()
                    progress.empty()

                    # Save info to session_state
                    st.session_state["processed_filename"] = (
                        uploaded_file.name
                    )
                    st.session_state["pdf_summary"] = (
                        get_pdf_summary(pages)
                    )
                    st.session_state["chunks_summary"] = (
                        get_chunks_summary(chunks)
                    )

                    st.success("✅ PDF processed successfully!")

                except ValueError as e:
                    st.error(f"❌ {str(e)}")

                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

        else:
            st.success(f"✅ {uploaded_file.name} is ready!")

    # Show document stats after processing
    if "pdf_summary" in st.session_state:
        st.divider()
        st.subheader("📊 Document Stats")

        pdf_info = st.session_state["pdf_summary"]
        chunks_info = st.session_state["chunks_summary"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Pages", pdf_info["total_pages"])
            st.metric("Chunks", chunks_info["total_chunks"])

        with col2:
            st.metric(
                "Characters",
                f"{pdf_info['total_characters']:,}"
            )
            st.metric(
                "Avg Chunk",
                f"{chunks_info['avg_chunk_size']}c"
            )

# ---------------------------------------------------------
# MAIN AREA — Q&A INTERFACE
# ---------------------------------------------------------
pdf_is_ready = (
    "processed_filename" in st.session_state or
    vector_store_exists()
)

if not pdf_is_ready:

    # Guide user to upload first
    st.markdown("""
    ### 👈 Get Started

    1. Open the **sidebar** on the left
    2. Upload your **HR policy PDF**
    3. Click **Process PDF**
    4. Start asking questions!

    ---
    **Example questions you can ask:**
    - *"What is the annual leave policy?"*
    - *"How many sick days am I entitled to?"*
    - *"What is the work from home policy?"*
    - *"What are the maternity leave benefits?"*
    """)

else:

    st.subheader("💬 Ask a Question")

    # Show which document is loaded
    if "processed_filename" in st.session_state:
        fname = st.session_state["processed_filename"]
        st.markdown(
            f'<div class="source-badge">📄 {fname}</div>',
            unsafe_allow_html=True
        )

    # Question input box
    question = st.text_input(
        label="Type your question here",
        placeholder="e.g. What is the casual leave policy?",
        label_visibility="collapsed"
    )

    # Ask and Clear buttons side by side
    col1, col2 = st.columns([3, 1])

    with col1:
        ask_clicked = st.button(
            "🔍 Ask",
            type="primary",
            use_container_width=True
        )

    with col2:
        clear_clicked = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    # Handle clear
    if clear_clicked:
        if "last_answer" in st.session_state:
            del st.session_state["last_answer"]
        st.rerun()

    # Handle ask
    if ask_clicked:

        if not question.strip():
            st.warning("⚠️ Please type a question first.")

        else:
            try:
                with st.spinner("🔍 Searching document..."):
                    vector_store = load_vector_store(
                        embedding_model
                    )
                    relevant_chunks = search_similar_chunks(
                        vector_store,
                        question
                    )

                with st.spinner("🤖 Generating answer..."):
                    result = answer_question(
                        question,
                        relevant_chunks
                    )

                # Store result so it survives reruns
                st.session_state["last_answer"] = {
                    "question": question,
                    "result": result
                }

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display answer if one exists
    if "last_answer" in st.session_state:

        data = st.session_state["last_answer"]
        result = data["result"]

        st.divider()
        st.subheader("📋 Answer")

        # Styled answer box
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True
        )

        # Source citation badge
        if result["source_pages"]:
            pages_str = ", ".join(
                f"Page {p}" for p in result["source_pages"]
            )
            st.markdown(
                f'<div class="source-badge">'
                f'📄 Source: {pages_str}</div>',
                unsafe_allow_html=True
            )

        # Expandable context viewer
        with st.expander("🔍 View retrieved context"):
            st.markdown(
                "*These are the exact document sections "
                "used to generate the answer:*"
            )
            st.text(result["context_used"])

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown(
    "<center><small>HR Policy Q&A Bot • "
    "Built with LangChain + Groq + FAISS + Streamlit"
    "</small></center>",
    unsafe_allow_html=True
)