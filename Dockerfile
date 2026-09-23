FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY agent ./agent
# static/ (the UI) is deliberately not copied: in Cloudflare the Worker
# serves it, and keeping it out of the image means UI-only deploys don't
# roll out a new container version (which replaces running containers and
# drops live calls).
COPY main.py ./

EXPOSE 7860

CMD ["uv", "run", "python", "main.py", "-t", "webrtc", "--host", "0.0.0.0"]
