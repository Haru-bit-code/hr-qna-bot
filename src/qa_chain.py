# src/qa_chain.py
# ---------------------------------------------------------
# QA Chain — updated to use Groq API
# Uses LCEL (LangChain Expression Language) pipe syntax
# ---------------------------------------------------------

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import (
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    MAX_TOKENS
)

# ---------------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------------
PROMPT_TEMPLATE = """
You are a helpful and professional HR policy assistant.
Your job is to answer employee questions based strictly
on the HR policy document provided.

IMPORTANT RULES:
- Answer ONLY using the information in the context below.
- If the answer is not found in the context, say:
  "I couldn't find information about this in the
   provided HR policy document."
- Do NOT make up information or use outside knowledge.
- Keep your answer clear, concise, and professional.
- If relevant, mention the specific policy section.

CONTEXT FROM HR DOCUMENT:
{context}

EMPLOYEE QUESTION:
{question}

YOUR ANSWER:
"""


def build_llm() -> ChatGroq:
    """
    Initializes and returns the Groq LLM client.

    Groq runs LLMs on custom LPU hardware —
    Language Processing Units — making inference
    extremely fast compared to standard GPU setups.

    Returns:
        ChatGroq: configured LLM ready to call
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please add it to your .env file."
        )

    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=GROQ_API_KEY,

        # temperature=0 means fully deterministic
        # Perfect for factual HR policy answers
        temperature=0,

        max_tokens=MAX_TOKENS
    )

    return llm


def format_context(retrieved_chunks: list) -> str:
    """
    Formats retrieved chunks into a labeled context string.

    Args:
        retrieved_chunks: list of Document objects from FAISS

    Returns:
        str: clean formatted context for the prompt
    """

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.metadata.get("source", "Unknown source")
        context_parts.append(
            f"[Excerpt {i+1} — {source}]\n"
            f"{chunk.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_source_pages(retrieved_chunks: list) -> list:
    """
    Extracts unique sorted page numbers from chunks.

    Args:
        retrieved_chunks: list of Document objects

    Returns:
        sorted list of unique page numbers
    """

    page_nums = set()

    for chunk in retrieved_chunks:
        page_num = chunk.metadata.get("page_num")
        if page_num:
            page_nums.add(page_num)

    return sorted(list(page_nums))


def answer_question(question: str,
                    retrieved_chunks: list) -> dict:
    """
    Core function: question + chunks → answer dict.

    Pipeline:
        PromptTemplate → ChatGroq → StrOutputParser

    Args:
        question: user's plain English question
        retrieved_chunks: top K docs from FAISS search

    Returns:
        dict with answer, source_pages, context_used
    """

    # Step 1 — Format chunks into readable context
    context = format_context(retrieved_chunks)

    # Step 2 — Initialize components
    llm = build_llm()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # StrOutputParser converts AIMessage object
    # returned by ChatGroq into a plain Python string
    output_parser = StrOutputParser()

    # Step 3 — Build LCEL chain
    # prompt fills the template →
    # llm generates the answer →
    # output_parser converts to string
    chain = prompt | llm | output_parser

    # Step 4 — Execute chain
    answer_text = chain.invoke({
        "context": context,
        "question": question
    })

    # Step 5 — Collect source citations
    source_pages = get_source_pages(retrieved_chunks)

    return {
        "answer": answer_text.strip(),
        "source_pages": source_pages,
        "context_used": context
    }