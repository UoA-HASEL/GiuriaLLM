FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

LABEL org.opencontainers.image.title="GiuriaLLM" \
      org.opencontainers.image.description="A configurable LLM-as-a-Judge framework for software engineering" \
      org.opencontainers.image.source="https://github.com/UoA-HASEL/GiuriaLLM" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

WORKDIR /opt/giuriallm
COPY requirements.txt ./
RUN pip install --require-hashes -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-deps . && rm -rf src
COPY demo/ ./demo/

RUN useradd --create-home --uid 1000 giuriallm \
    && mkdir /work && chown giuriallm:giuriallm /work
USER giuriallm
WORKDIR /work

ENTRYPOINT ["giuriallm"]
CMD ["--help"]
