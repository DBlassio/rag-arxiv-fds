FROM python:3.11-slim

WORKDIR /var/task

ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --timeout 120 \
    --retries 5 \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    --timeout 120 \
    --retries 5 \
    -r requirements.txt

RUN pip install --no-cache-dir awslambdaric


# ---------------------------------------------------------------------------
# Model storage
# ---------------------------------------------------------------------------

RUN mkdir -p /opt/models

# Download exact model files during image build.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(\
    repo_id='BAAI/bge-small-en-v1.5', \
    local_dir='/opt/models/bge-small-en-v1.5'\
)"

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(\
    repo_id='BAAI/bge-reranker-v2-m3', \
    local_dir='/opt/models/bge-reranker-v2-m3'\
)"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

COPY src/ ./src/
COPY app/ ./app/
COPY chroma_db/ ./chroma_db/


# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------

ENTRYPOINT ["python", "-m", "awslambdaric"]

CMD ["app.main.handler"]