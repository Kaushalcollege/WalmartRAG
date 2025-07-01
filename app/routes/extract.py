from fastapi import APIRouter, UploadFile, File
from app.services.pdf_handler import parse_and_chunk_pdf
from app.services.vector_store import store_to_vector_db_and_query

router = APIRouter()

@router.post("/extract")
async def extract_from_pdf(
    file: UploadFile = File(...),
    query: str = "Extract product metadata"
):
    # Step 1: Read and parse the PDF
    documents = await parse_and_chunk_pdf(file)

    # Step 2: Pass parsed docs to vector store and query
    response = store_to_vector_db_and_query(documents, query)

    return {"response": response}
