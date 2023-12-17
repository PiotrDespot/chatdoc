import asyncio
import itertools
import numpy as np
from sqlalchemy.sql import select, text, literal, desc
from sqlalchemy.sql import func
from app.postgres.tables import Chunks
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession


class SearchType(Enum):
    HYBRID = 'hybrid'
    FULL_TEXT = 'full_text'
    SEMANTIC = 'semantic'
    RRF = 'rrf'


class Search:

    @classmethod
    async def __rank_bm25(cls, session: AsyncSession, query: str):
        statement = select(Chunks.chunk_id, Chunks.content, func.paradedb.__rank_bm25(text("ctid")))\
            .where(Chunks.content.op("@@@")(literal(query)))

        response = await session.execute(statement)
        return response.all()

    @classmethod
    async def __semantic_search(cls, session: AsyncSession, vector: np.ndarray, limit: int):
        statement = select(Chunks.chunk_id, Chunks.content)\
            .order_by(Chunks.embeddings.cosine_distance(vector))\
            .limit(limit)

        response = await session.execute(statement)
        return response.all()

    @classmethod
    async def __ts_vector_rank(cls, session: AsyncSession, query: str, limit: int):
        plainto_tsquery = func.plainto_tsquery('english', query)
        ts_rank_cd = func.ts_rank_cd(func.to_tsvector('english', Chunks.content), plainto_tsquery)

        statement = select(Chunks.chunk_id, Chunks.content)\
            .select_from(plainto_tsquery)\
            .where(func.to_tsvector('english', Chunks.content).op("@@")(plainto_tsquery))\
            .order_by(ts_rank_cd.desc())\
            .limit(limit)

        response = await session.execute(statement)
        return response.all()

    @classmethod
    async def __hybrid_search(cls, session: AsyncSession, query: str, vector: np.ndarray, limit: int):
        response = await asyncio.gather(cls.__semantic_search(session, vector, limit), cls.__ts_vector_rank(session, query, limit))
        return set(itertools.chain(*response))

    @classmethod
    async def __reciprocal_rank_fusion(cls, session: AsyncSession, query: str, vector: np.ndarray, limit: int):
        rank_label = "rank"

        def coalesce_divide(other):
            return func.coalesce(literal(1.0).op("/")(literal(60).op("+")(other)), literal(0.0))

        semantic_search = select(
            Chunks.chunk_id,
            Chunks.content,
            func.rank().over(order_by=Chunks.embeddings.cosine_distance(vector)).label(rank_label)).order_by(
            Chunks.embeddings.cosine_distance(vector)).limit(limit).cte()

        plainto_tsquery = func.plainto_tsquery('english', query)

        keyword_search = select(Chunks.chunk_id, Chunks.content, func.rank().over(order_by=func.ts_rank_cd(
            func.to_tsvector('english', Chunks.content), plainto_tsquery).desc()).label(rank_label))\
            .select_from(plainto_tsquery)\
            .where(func.to_tsvector('english', Chunks.content).op("@@")(plainto_tsquery))\
            .order_by(desc(func.ts_rank_cd(func.to_tsvector('english', Chunks.content), plainto_tsquery)))\
            .limit(limit)\
            .cte()

        statement = select(
            func.coalesce(semantic_search.c.chunk_id, keyword_search.c.chunk_id).label("id"),
            func.coalesce(semantic_search.c.content, keyword_search.c.content).label("content"))\
            .outerjoin(keyword_search, onclause=semantic_search.c.chunk_id.op("=")(keyword_search.c.chunk_id), full=True)\
            .order_by(desc(coalesce_divide(semantic_search.c.rank).op("+")(coalesce_divide(keyword_search.c.rank))))\
            .limit(limit)

        response = await session.execute(statement)
        return response.all()

    @classmethod
    async def search(cls, session: AsyncSession, query: str, vector: np.ndarray, limit: int, search_type: SearchType):
        match search_type:
            case SearchType.HYBRID:
                return await cls.__hybrid_search(session, query, vector, limit)

            case SearchType.FULL_TEXT:
                return await cls.__ts_vector_rank(session, query, limit)

            case SearchType.RRF:
                return await cls.__reciprocal_rank_fusion(session, query, vector, limit)

            case SearchType.SEMANTIC:
                return await cls.__semantic_search(session, vector, limit)
