import os

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

question = "What is my rank?"

results = db.similarity_search(
    question,
    k=3
)

context = "\n\n".join(
    [doc.page_content for doc in results]
)

prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(response.content)