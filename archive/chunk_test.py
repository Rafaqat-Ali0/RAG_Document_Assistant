from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("sample.pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Number of Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0].page_content)

if len(chunks) > 1:
    print("\n--------------------\n")
    print("Second Chunk:\n")
    print(chunks[1].page_content)