import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.loader import PDFLoader
from src.chunker import TextChunker
from src.vector_store import VectorStore
from src.rag_chain import RAGChain

load_dotenv(override=True)

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)

# =========================
# HEADER
# =========================

st.title("📄 RAG Document Assistant")

st.markdown(
    """
    Ask questions about PDF documents using Retrieval-Augmented Generation (RAG).
    """
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📄 Project Information")

    st.write("RAG Document Assistant")

    st.write("LangChain + Gemini + ChromaDB")

    st.write("Upload a PDF and ask questions.")

    st.markdown("---")

    st.write("Version 1.0")

    st.write("Single PDF RAG System")

# =========================
# SESSION STATE
# =========================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# CLEAR CHAT
# =========================

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# =========================
# PDF PROCESSING
# =========================

if uploaded_file and st.session_state.vector_db is None:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.read()
            )

            pdf_path = tmp_file.name

        documents = PDFLoader.load_pdf(
            pdf_path
        )

        chunks = TextChunker.create_chunks(
            documents
        )

        vector_db = VectorStore.create_vector_db(
            chunks
        )

        st.session_state.vector_db = vector_db

        st.success(
            f"✅ {uploaded_file.name} processed successfully"
        )

        st.info(
            f"""
📄 File: {uploaded_file.name}

📑 Pages: {len(documents)}

🧩 Chunks: {len(chunks)}
"""
        )

# =========================
# METRICS
# =========================

if st.session_state.vector_db:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Questions Asked",
            len(st.session_state.chat_history)
        )

    with col2:
        st.metric(
            "Retriever k",
            8
        )

    with col3:
        st.metric(
            "Status",
            "Ready"
        )

# =========================
# QUESTION INPUT
# =========================

if st.session_state.vector_db:

    question = st.text_input(
        "Ask a question about the document"
    )

    if st.button("Get Answer"):

        if question.strip():

            try:

                results = (
                    st.session_state.vector_db
                    .similarity_search(
                        question,
                        k=8
                    )
                )

                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in results
                    ]
                )

                pages = []

                for doc in results:

                    if "page" in doc.metadata:

                        pages.append(
                            doc.metadata["page"] + 1
                        )

                answer = (
                    RAGChain.answer_question(
                        context,
                        question
                    )
                )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "pages": sorted(
                            list(set(pages))
                        ),
                        "context": context
                    }
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )

# =========================
# CHAT HISTORY
# =========================

for chat in reversed(
    st.session_state.chat_history
):

    st.markdown("---")

    st.markdown(
        f"### ❓ {chat['question']}"
    )

    st.markdown(
        "### 🤖 Answer"
    )

    st.info(
        chat["answer"]
    )

    st.success(
        f"📚 Source Pages: {chat['pages']}"
    )

    with st.expander(
        "🔍 Retrieved Context"
    ):

        st.write(
            chat["context"]
        )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Built with LangChain • Gemini • ChromaDB • Streamlit"
)