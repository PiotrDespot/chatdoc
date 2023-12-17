FROM tiangolo/uvicorn-gunicorn:python3.10

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0


# Install the package
RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config
RUN python -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

RUN apt-get update && apt-get install -y curl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DBUILD_SHARED_LIBS=ON" pip install llama_cpp_python --verbose

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt


COPY env_template .env
COPY ./app ./app

RUN mkdir -p ./models
#COPY ./models ./models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]



