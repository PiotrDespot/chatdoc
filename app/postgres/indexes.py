from pydantic import BaseModel
from sqlalchemy import Index, literal
from sqlalchemy.engine.base import Engine
from typing import Optional
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import func


class _PgIndex(BaseModel, arbitrary_types_allowed=True):
    _name: str
    index_name: str
    table_column: InstrumentedAttribute
    ops: Optional[str] = None

    def create(self, engine: Engine) -> None:
        pg_vars = {"postgresql_using": self._name}

        if self._name == "hnsw":
            pg_vars["postgresql_ops"] = {self.table_column.key: self.ops}
        elif self._name == "bm25":
            pg_vars["postgresql_with"] = {"text_fields": """'{%s: {tokenizer: {type: "en_stem"}}}'""" % self.table_column.key}

        self.index(pg_vars).create(engine, checkfirst=True)

    def index(self, pg_vars: dict) -> Index:
        return Index(self.index_name, self.table_column, **pg_vars)


class HnswIndex(_PgIndex):
    _name: str = "hnsw"


class Bm25Index(_PgIndex):
    _name: str = "bm25"


class GinIndex(_PgIndex):
    _name: str = "GIN"

    def index(self, pg_vars: dict) -> Index:
        return Index(self.index_name, func.to_tsvector(literal('english'), self.table_column), **pg_vars)
