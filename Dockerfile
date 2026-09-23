FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY agent ./agent
COPY static ./static
COPY main.py ./

EXPOSE 7860

CMD ["uv", "run", "python", "main.py", "-t", "webrtc", "--host", "0.0.0.0"]
