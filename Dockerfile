FROM python:3.13-slim

# Install system dependencies (ffmpeg and mkvtoolnix)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    mkvtoolnix \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create required directories
RUN mkdir -p /app/app/static /app/data

ARG BABEL_VERSION
ENV BABEL_VERSION=${BABEL_VERSION}
ENV PYTHONUNBUFFERED=1
ENV PORT=8765

EXPOSE 8765

# Bug #44: Remove --reload for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
