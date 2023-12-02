import asyncio
import io
import os
from typing import Annotated

from anyio.streams.memory import MemoryObjectSendStream

from app.confluence_importer import ConfluenceImporter
from fastapi import APIRouter, Body, status, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from pypdf import PdfReader
from pgml import Collection, Model, Pipeline, Splitter
from FlagEmbedding import FlagReranker
from pydantic import BaseModel
from markdown import markdown
from bs4 import BeautifulSoup
from starlette.concurrency import iterate_in_threadpool
from llama_cpp import Llama
from sse_starlette.sse import EventSourceResponse
from functools import partial
from typing import Iterator
import numpy as np
import json
import anyio

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


router = APIRouter(
    prefix="/collections",
    tags=["collections"],
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


global llm


@router.on_event("startup")
def llm_init():
    global llm
    llm = Llama(model_path="./models/mistral-7b-openorca.Q4_K_M.gguf", chat_format='chatml', n_ctx=2048)


@router.on_event("shutdown")
def llm_del():
    global llm
    del llm


@router.post("/documents/upsert")
async def upsert_documents(collection_name: str, documents: Annotated[list[dict], Body(examples=example_documents)]) -> None:
    collection = Collection(collection_name)
    return await collection.upsert_documents(documents)


@router.post("/documents/upload")
async def upload_document(
    collection_name: Annotated[str, Form()],
    file_type: Annotated[str, Form()],
    document: Annotated[UploadFile, File()]
):
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
    elif file_type == "pdf":
        reader = PdfReader(io.BytesIO(contents))
        for page in reader.pages:
            documents = [{
                "id": document.filename,
                "text": page.extract_text(),
            }]
            await collection.upsert_documents(documents)
    elif file_type == "confluence":

        confluence_params = {
            "url": "your_confluence_url",
            "username": "your_confluence_username",
            "token": "your_confluence_token",
            "out_dir": "your_output_directory",
            "space": "your_confluence_space",
            "no_attach": False,  # Set to True or False based on your requirements
            "no_fetch": False,   # Set to True or False based on your requirements
        }

        # Create an instance of ConfluenceExporter and run it
        exporter = ConfluenceImporter(confluence_params)
        exporter.run()

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
    print(relevant_document)

    message = llm.create_chat_completion(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": user_message.format(query=prompt.query, context=relevant_document)
            }
        ],
        stream=True,
        stop="<|im_end|>"
    )

    send_channel, receive_channel = anyio.create_memory_object_stream()
    return EventSourceResponse(receive_channel, data_sender_callable=partial(
        get_message_iterator,
        send_channel=send_channel,
        iterator=message,

    ))


async def get_message_iterator(send_channel: MemoryObjectSendStream, iterator: Iterator):
    async with send_channel:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                await send_channel.send(dict(data=json.dumps(chunk)))
        except anyio.get_cancelled_exc_class() as e:
            print("disconneced")
            with anyio.move_on_after(1, shield=True):
                raise e


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
    models_data = []

    model_paths = ["mistral:latest", "/path/to/model2"]

    for model_path in model_paths:
        model_name = model_path
        modified_at = datetime.now().isoformat()
        size = 10000

        model_info = {
            "name": model_name,
            "modified_at": modified_at,
            "size": size
        }

        models_data.append(model_info)

    response_data = {"models": models_data}
    return JSONResponse(content=response_data)


def get_model_details(model_name):
    # Example model path, replace this with your logic to fetch the actual model details
    model_path = f"/path/to/{model_name}"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    modified_at = datetime.utcfromtimestamp(os.path.getmtime(model_path)).isoformat()
    size = os.path.getsize(model_path)

    # Replace the following placeholders with your logic to fetch specific model details
    license_content = "<contents of license block>"
    modelfile_content = "# Modelfile content"
    parameters_content = "stop                           [INST]\nstop                           [/INST]\nstop                           <<SYS>>\nstop                           <</SYS>>"
    template_content = "[INST] {{ if and .First .System }}<<SYS>>{{ .System }}<</SYS>>\n\n{{ end }}{{ .Prompt }} [/INST] "

    model_details = {
        "license": license_content,
        "modelfile": modelfile_content,
        "parameters": parameters_content,
        "template": template_content,
        "modified_at": modified_at,
        "size": size
    }

    return model_details

@router.post("/show")
async def show_model(name: str):
    try:
        model_details = get_model_details(name)
        return JSONResponse(content=model_details)
    except HTTPException as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.status_code)

