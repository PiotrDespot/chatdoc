from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.postgres.indexes import HnswIndex, Bm25Index, GinIndex
from app.postgres.tables import BaseTable, Chunks
from app.config import get_settings


settings = get_settings()
engine = create_async_engine(settings.database_url)


async def init_db():
    async with engine.begin() as conn:
        # await conn.run_sync(BaseTable.metadata.drop_all)
        await conn.run_sync(BaseTable.metadata.create_all)

        if settings.hnsw_index_enabled:
            await conn.run_sync(HnswIndex(index_name="hnsw", table_column=Chunks.embeddings, ops="vector_cosine_ops").create)

        if settings.bm25_index_enabled:
            await conn.run_sync(Bm25Index(index_name="bm25", table_column=Chunks.content).create)
        else:
            await conn.run_sync(GinIndex(index_name="gin", table_column=Chunks.content).create)


async def get_session():
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        yield session
