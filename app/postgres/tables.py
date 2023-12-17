from sqlalchemy.orm import DeclarativeBase, mapped_column, relationship
from sqlalchemy import VARCHAR, TIMESTAMP, TEXT, BigInteger, ForeignKey
from sqlalchemy.ext.asyncio import AsyncAttrs
from app.postgres.vector import Vector


class BaseTable(AsyncAttrs, DeclarativeBase):
    pass


class Documents(BaseTable):
    __tablename__ = "documents"

    id = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source = mapped_column(VARCHAR)
    path = mapped_column(VARCHAR)
    permissions = mapped_column(VARCHAR)
    created_at = mapped_column(TIMESTAMP)

    chunks = relationship("Chunks", back_populates="documents")


class Chunks(BaseTable):
    __tablename__ = "chunks"

    chunk_id = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id = mapped_column(BigInteger, ForeignKey(Documents.id), nullable=False)
    content = mapped_column(TEXT, nullable=False)
    embeddings = mapped_column(Vector(768), nullable=False)
    chunk_type = mapped_column(VARCHAR, nullable=True)
    created_at = mapped_column(TIMESTAMP, nullable=False)

    documents = relationship("Documents", back_populates="chunks")


class Users(BaseTable):
    __tablename__ = "users"

    id = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username = mapped_column(VARCHAR)
    permissions = mapped_column(VARCHAR)
    created_at = mapped_column(TIMESTAMP)
