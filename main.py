from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import extract

app = FastAPI(
    title="Walmart AI Retail",
    description="Auto-extract product listings from PDFs/Docs using AWS Bedrock + LLMs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract.router)

@app.get("/")
def read_root():
    return {"message": "Walmart AI Co-Pilot"}
