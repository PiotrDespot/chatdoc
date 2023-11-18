import asyncio
import io
from typing import Annotated

from fastapi import APIRouter, Body, status, UploadFile, Form, File
from pypdf import PdfReader
from pgml import Collection, Model, Pipeline, Splitter
from FlagEmbedding import FlagReranker
from pydantic import BaseModel
from markdown import markdown
from bs4 import BeautifulSoup
from ctransformers import AutoModelForCausalLM, LLM, AutoConfig, Config
from fastapi.responses import StreamingResponse
import numpy as np
import json

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

user_message = """
Based on documents listed below answer the following question: {query}
###
Documents
###
{context}
"""

chatbot_input_string = """<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant"""


router = APIRouter(
    prefix="/collections",
    tags=["collections"]
)


class RelevantDocument(BaseModel):
    query: str
    content: str


class Prompt(BaseModel):
    collection_name: str
    pipeline_name: str
    query: str
    limit: int


class PromptStreamingResponse(BaseModel):
    response: str


@router.post("/documents/upsert")
async def upsert_documents(collection_name: str, documents: Annotated[list[dict], Body(examples=example_documents)]) -> None:
    collection = Collection(collection_name)
    return await collection.upsert_documents(documents)


@router.post("/documents/upload")
async def upload_document(collection_name: Annotated[str, Form()], file_type: Annotated[str, Form()], document: Annotated[UploadFile, File()]):
    collection = Collection(collection_name)
    contents = await document.read()
    if file_type == "markdown":
        html = markdown(contents)
        text = ''.join(BeautifulSoup(html, features='html.parser').find_all(text=True))
        documents = [{
            "id": document.filename,
            "text": str(text),
        }]

        await collection.upsert_documents(documents)
    else:
        reader = PdfReader(io.BytesIO(contents))
        for page in reader.pages:
            documents = [{
                "id": document.filename,
                "text": page.extract_text(),
            }]
            await collection.upsert_documents(documents)

    return status.HTTP_200_OK


@router.get("/documents/retrieve")
async def retrieve_documents(collection_name: str):
    collection = Collection(collection_name)
    return await collection.get_documents({"limit": 100})


@router.post("/documents/embed")
async def embed(collection_name: str, pipeline_name: str = "embedding_pipeline", model_name: str = "BAAI/bge-base-en-v1.5"):
    collection = Collection(collection_name)
    model = Model(model_name, source="pgml")
    splitter = Splitter()
    pipeline = Pipeline(pipeline_name, model, splitter)
    return await collection.add_pipeline(pipeline)


@router.post("/documents/query")
async def query_documents(collection_name: str, query: str, pipeline_name: str, limit: int):
    return await find_most_relevant_document(collection_name, query, pipeline_name, limit)


@router.delete("/archive")
async def archive_collection(collection_name: str):
    collection = Collection(collection_name)
    return await collection.archive()


@router.post("/chat/query")
async def chat_with_model(prompt: Prompt):
    relevant_document = await find_most_relevant_document(prompt.collection_name, prompt.pipeline_name, prompt.query, prompt.limit)
    formatted_user_message = user_message.format(query=prompt.query, context=relevant_document)
    model_input_string = chatbot_input_string.format(user_message=formatted_user_message)

    config = Config(max_new_tokens=256, context_length=3000, stop=["<|im_end|>"])
    auto_config = AutoConfig(config=config)
    model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF", model_file="mistral-7b-openorca.Q4_K_M.gguf", model_type="mistral", gpu_layers=0, config=auto_config)
    return StreamingResponse(chat_model_response_streamer(model, model_input_string), media_type="application/json")


async def chat_model_response_streamer(model: LLM, query: str):
    for chunk in model(query):
        yield json.dumps({"response": chunk})
        await asyncio.sleep(0.1)


async def find_most_relevant_document(collection_name: str, pipeline_name: str, query: str, limit: int):
    collection = Collection(collection_name)
    pipeline = await collection.get_pipeline(pipeline_name)
    results = await collection.query().vector_recall(query, pipeline).limit(limit).fetch_all()

    reranker = FlagReranker('BAAI/bge-reranker-base')
    mapped_results: list[tuple[str, str]] = [(query, result[1]) for result in results]
    scores = reranker.compute_score(mapped_results)
    best_doc = np.argmax(scores)
    return results[best_doc][1]


@router.get("/models")
async def get_models():
    return "mistral"
