from langchain.chains import RetrievalQA
from langchain_community.llms import Bedrock

def build_rag_chain(vectorstore, model_id="amazon.titan-text-lite-v1"):
    retriever = vectorstore.as_retriever()

    llm = Bedrock(
        client=boto3.client("bedrock-runtime", region_name="ap-south-1"),
        model_id=model_id,
        model_kwargs={"temperature": 0.3, "maxTokenCount": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain