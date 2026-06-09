from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:

    @staticmethod
    def create_chunks(documents):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        return splitter.split_documents(documents)