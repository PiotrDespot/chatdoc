from fastapi import FastAPI
from dotenv import load_dotenv
from app.routers import collection_v1
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
app.include_router(collection_v1.router, prefix="/v1")
