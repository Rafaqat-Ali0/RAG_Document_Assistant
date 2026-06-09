import os

from langchain_google_genai import ChatGoogleGenerativeAI


class RAGChain:

    @staticmethod
    def answer_question(context, question):

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        prompt = f"""
You are a document assistant.

Answer only using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        return response.content