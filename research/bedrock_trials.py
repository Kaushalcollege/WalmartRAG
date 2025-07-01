# research/bedrock_trials.py

from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import boto3


def get_langchain_bedrock_llm():
    # Use correct model ID — without ":0:4k"
    model_id = "amazon.titan-text-lite-v1"
    client = boto3.client("bedrock-runtime", region_name="ap-south-1")

    return Bedrock(
        client=client,
        model_id=model_id,
        model_kwargs={
            "temperature": 0.3,
            "maxTokenCount": 512,
            "topP": 0.9
        }
    )


def run_llm_chain(product_description: str):
    prompt_template = PromptTemplate(
        input_variables=["product_description"],
        template="""
You are an AI assistant. Extract the following fields from the given product description:
- product_title
- brand
- key_features (as a list)
- GTIN (if present)

Provide the result as JSON.

Product Description:
{product_description}
"""
    )

    llm = get_langchain_bedrock_llm()
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run({"product_description": product_description})


if __name__ == "__main__":
    doc = """
Product: FitZone SmartBand
Features: Sleep Tracker, Heart Rate Monitor, Bluetooth, Waterproof
GTIN: 123456789012
Brand: FitZone
"""
    result = run_llm_chain(doc)
    print("🔍 Result from Titan:\n", result)
