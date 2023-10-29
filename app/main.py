from fastapi import FastAPI
from dotenv import load_dotenv
from app.routers import collection_v1

app = FastAPI()
load_dotenv()
app.include_router(collection_v1.router, prefix="/v1")
