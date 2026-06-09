from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStore:

    @staticmethod
    def create_vector_db(chunks):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        return db