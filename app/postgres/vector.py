import numpy as np
from sqlalchemy import Dialect
from sqlalchemy.types import UserDefinedType, Float
from sqlalchemy.dialects.postgresql.base import ischema_names


class Vector(UserDefinedType):
    cache_ok = True

    def __init__(self, dim=None):
        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kwargs):
        if self.dim is None:
            return "VECTOR"

        return f"VECTOR({self.dim})"

    def bind_processor(self, dialect: Dialect):
        def to_database(value):
            if value is None:
                return value

            value_length = len(value)
            if self.dim is not None and value_length != self.dim:
                raise ValueError(f"Expected {self.dim} dimensions, but got {value_length}")

            if isinstance(value, np.ndarray):
                if value.ndim != 1:
                    raise ValueError('Expected dimension to be 1')

                if not np.issubdtype(value.dtype, np.integer) and not np.issubdtype(value.dtype, np.floating):
                    raise ValueError('dtype must be numeric')

                value = value.tolist()

            return f"[{','.join([str(float(v)) for v in value])}]"

        return to_database

    def result_processor(self, dialect: Dialect, coltype: object):
        def from_database(value):
            if value is None or isinstance(value, np.ndarray):
                return value

            return np.array(value[1:-1].split(','), dtype=np.float32)

        return from_database

    class comparator_factory(UserDefinedType.Comparator):
        def cosine_distance(self, other):
            return self.op('<=>', return_type=Float)(other)

        def l2_distance(self, other):
            return self.op('<->', return_type=Float)(other)

        def inner_product(self, other):
            return self.op("<#>", return_type=Float)(other)


ischema_names['vector'] = Vector
