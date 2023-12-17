from fastapi import FastAPI
from app.routers import collection_v1, documents, information
from app.postgres.db import init_db
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(information.router)
# app.include_router(collection_v1.router) deprecated


@app.on_event("startup")
async def on_startup():
    await init_db()


@app.post("/")
async def redirect():
    return RedirectResponse("/docs")
