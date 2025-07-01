from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile

async def parse_and_chunk_pdf(file):
    # Save uploaded file to a temp file
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(contents)
        temp_pdf_path = temp_pdf.name

    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load_and_split()

    return documents
