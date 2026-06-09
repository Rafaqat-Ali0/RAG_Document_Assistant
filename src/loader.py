from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:

    @staticmethod
    def load_pdf(pdf_path):

        loader = PyPDFLoader(pdf_path)

        return loader.load()