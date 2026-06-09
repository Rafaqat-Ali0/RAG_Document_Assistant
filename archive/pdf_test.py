from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")

documents = loader.load()

print("Number of pages:", len(documents))

print("\nFirst page content:\n")

print(documents[0].page_content[:1000])