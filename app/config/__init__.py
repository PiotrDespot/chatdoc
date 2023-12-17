from functools import lru_cache
from app.postgres.search import SearchType
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    bm25_index_enabled: bool
    hnsw_index_enabled: bool
    document_search: SearchType

    confluence_url: str
    confluence_username: str
    confluence_token: str
    confluence_out_dir: str
    confluence_space: str
    no_attach: bool
    no_fetch: bool

    model_config = SettingsConfigDict(env_file="/home/nbpdespotmladanowicz/personal-llm-projects/chatdoc/.env")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
