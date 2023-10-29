from typing import Annotated

from fastapi import APIRouter, Body, status, UploadFile
from pgml import Collection, Model, Pipeline, Splitter
from FlagEmbedding import FlagReranker
from pydantic import BaseModel
from markdown import markdown
from bs4 import BeautifulSoup
from ctransformers import AutoModelForCausalLM


example_documents = [
        {
            "id": "Document 1",
            "text": "Here are the contents of Document 1",
            "random_key": "this will be metadata for the document"
        },
        {
            "id": "Document 2",
            "text": "Here are the contents of Document 2",
            "random_key": "this will be metadata for the document"
        }
    ]

router = APIRouter(
    prefix="/collections",
    tags=["collections"]
)


class RelevantDocument(BaseModel):
    query: str
    content: str


@router.post("/documents/upsert")
async def upsert_documents(collection_name: str, documents: Annotated[list[dict], Body(examples=example_documents)]) -> bool:
    collection = Collection(collection_name)
    await collection.upsert_documents(documents)
    is_created = await collection.exists()
    return status.HTTP_201_CREATED if is_created else status.HTTP_417_EXPECTATION_FAILED


@router.post("/documents/upload")
async def upload_document(collection_name: str, document: UploadFile):
    collection = Collection(collection_name)
    contents = await document.read()
    html = markdown(contents)
    text = ''.join(BeautifulSoup(html, features='html.parser').find_all(text=True))
    documents = [{
        "id": document.filename,
        "text": str(text),
    }]
    await collection.upsert_documents(documents)
    return status.HTTP_200_OK


@router.get("/documents/retrieve")
async def retrieve_documents(collection_name: str):
    collection = Collection(collection_name)
    documents = await collection.get_documents({"limit": 100})
    return documents


@router.post("/documents/embed")
async def embed(collection_name: str, pipeline_name: str = "embedding_pipeline", model_name: str = "BAAI/bge-base-en-v1.5"):
    collection = Collection(collection_name)
    model = Model(model_name, source="pgml")
    splitter = Splitter()
    pipeline = Pipeline(pipeline_name, model, splitter)
    return await collection.add_pipeline(pipeline)


@router.post("/documents/query")
async def query_documents(collection_name: str, query: str, pipeline_name: str, limit: int):
    collection = Collection(collection_name)
    pipeline = await collection.get_pipeline(pipeline_name)
    results = await collection.query().vector_recall(query, pipeline).limit(limit).fetch_all()

    reranker = FlagReranker('BAAI/bge-reranker-base')
    mapped_results: list[tuple[str, str]] = [(query, result[1])for result in results]
    scores = reranker.compute_score(mapped_results)
    return zip(scores, results)


@router.delete("/archive")
async def archive_collection(collection_name: str):
    collection = Collection(collection_name)
    return await collection.archive()


@router.post("/chat/query")
async def chat_with_model():
    select_string = \
        """"<|im_start|>system
        You are a helpful assistant<|im_end|>
        <|im_start|>user
        Can you introduce yourself?<|im_end|>
        <|im_start|>assistant"""
    model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF", model_file="mistral-7b-openorca.Q4_K_M.gguf", model_type="mistral", gpu_layers=0)
    return model(select_string, max_new_tokens=256, temperature=0.8, repetition_penalty=1.1, stream=True)
