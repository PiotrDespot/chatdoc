import asyncio
import json
from functools import partial

import anyio
import torch
import numpy as np
from fastapi import APIRouter, Form, UploadFile, File, Depends, status
from fastapi.responses import JSONResponse
from typing import Annotated, List, Iterator
from app.postgres.tables import Chunks, Documents
from sentence_transformers import SentenceTransformer
from datetime import datetime
from app.postgres.search import Search
from app.dependencies import AppSettings, DatabaseSession
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from app.documents.document_uploader import DocumentFormats, MarkdownExtractor, ConfluenceExtractor, PdfExtractor
from llama_cpp import Llama
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import iterate_in_threadpool
from sse_starlette.sse import EventSourceResponse


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


embedding_model: SentenceTransformer | None = None
embedding_model_pool: ProcessPoolExecutor | None = None

reranker_tokenizer: PreTrainedTokenizerBase | None = None
reranker: SentenceTransformer | None = None

llm: Llama | None = None
llm_pool: ProcessPoolExecutor | None = None


def create_embedders():
    global reranker_tokenizer
    global reranker
    global embedding_model

    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    embedding_model.eval()
    embedding_model.share_memory()

    reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    reranker = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    reranker.eval()
    reranker.share_memory()


def embed(text_chunks: list[str]):
    vector = embedding_model.encode(text_chunks, normalize_embeddings=True)
    return vector


def rerank(query, chunks: list[int, str]):
    input_pairs = [[query, chunk] for idx, chunk in chunks]
    with torch.no_grad():
        inputs = reranker_tokenizer(input_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker(**inputs, return_dict=True).logits.view(-1, ).float()

    reranked_results = np.argsort(scores.numpy())[::-1]

    return np.array(chunks)[reranked_results]


@router.on_event("startup")
def startup():
    global embedding_model_pool
    global llm
    embedding_model_pool = ProcessPoolExecutor(max_workers=1, initializer=create_embedders)
    llm = Llama(model_path="./models/mistral-7b-openorca.Q4_K_M.gguf", chat_format='chatml', n_ctx=2048)


@router.on_event("shutdown")
def shutdown():
    global embedding_model_pool
    embedding_model_pool.shutdown(wait=True)


@router.post("/upload")
async def upload_document(
    file_type: Annotated[DocumentFormats, Form()],
    documents: Annotated[List[UploadFile], File()],
    session: DatabaseSession,
    settings: AppSettings
):
    match file_type:
        case DocumentFormats.MARKDOWN:
            extractor = MarkdownExtractor
        case DocumentFormats.PDF:
            extractor = PdfExtractor
        case DocumentFormats.CONFLUENCE:
            confluence_params = {
                "url": settings.confluence_url,
                "username": settings.confluence_username,
                "token": settings.confluence_token,
                "out_dir": settings.confluence_out_dir,
                "space": settings.confluence_space,
                "no_attach": settings.no_attach,  # Set to True or False based on your requirements
                "no_fetch": settings.no_fetch,  # Set to True or False based on your requirements
            }
            extractor = ConfluenceExtractor(confluence_settings=confluence_params)
        case _:
            return status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    # TODO: Add automatic file type matching and validation based on document.content_type
    loop = asyncio.get_event_loop()

    for document in documents:
        contents = await document.read()
        text = extractor.extract(contents)
        chunks = extractor.fixed_size_text_chunking(text)
        embeddings = await loop.run_in_executor(embedding_model_pool, embed, chunks)

        document_row = Documents(source=file_type.value, path=document.filename, permissions="public", created_at=datetime.now())
        session.add(document_row)
        await session.flush()

        rows_to_upload = [
            Chunks(
                doc_id=document_row.id,
                content=content,
                embeddings=embedding,
                chunk_type=file_type.value,
                created_at=datetime.now()
            )

            for content, embedding in zip(chunks, embeddings)
        ]

        session.add_all(rows_to_upload)
        await session.commit()

    return status.HTTP_200_OK


async def find_relevant_documents(
        query: str,
        session: DatabaseSession,
        settings: AppSettings
):
    loop = asyncio.get_event_loop()

    embedding = await loop.run_in_executor(embedding_model_pool, embed, query)
    result = await Search.search(session, query, embedding, 10, settings.document_search)
    reranked_result = await loop.run_in_executor(embedding_model_pool, rerank, query, result)

    response = [{"id": idx, "content": content} for idx, content in reranked_result]
    return response


@router.post("/query")
async def query_documents(relevant_documents: Annotated[dict, Depends(find_relevant_documents)]):
    return JSONResponse(relevant_documents)


async def get_message_iterator(send_channel: MemoryObjectSendStream, iterator: Iterator):
    async with send_channel:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                await send_channel.send(dict(data=json.dumps(chunk)))
        except anyio.get_cancelled_exc_class() as e:
            with anyio.move_on_after(1, shield=True):
                raise e


@router.post("/chat")
async def converse_with_llm(query: str, relevant_documents: Annotated[dict, Depends(find_relevant_documents)]):
    top_5_chunks = [chunk for result in relevant_documents for chunk in result.values()][:5]

    context = "\n###\n".join(top_5_chunks)

    user_message = \
        """
        Based on documents listed below answer the following question: {query}
        ###
        Documents
        ###
        {context}
        """

    message = llm.create_chat_completion(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": user_message.format(query=query, context=context)
            }
        ],
        stream=True,
        stop="<|im_end|>"
    )

    send_channel, receive_channel = anyio.create_memory_object_stream()

    return EventSourceResponse(receive_channel, data_sender_callable=partial(
        get_message_iterator,
        send_channel=send_channel,
        iterator=message
    ))
