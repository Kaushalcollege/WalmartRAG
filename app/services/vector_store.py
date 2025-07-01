from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def store_to_vector_db_and_query(documents, query):
    # Use text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(documents)

    # Embeddings using Bedrock Titan
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        region_name="ap-south-1",
        model_id="amazon.titan-embed-text-v2:0"
    )

    # Create FAISS vector store
    db = FAISS.from_documents(split_docs, embeddings)

    # Query the vector store
    response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in response] if response else ["No relevant content found."]
