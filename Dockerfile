FROM python:3.11-slim

WORKDIR /app

# Image processing dependencies (libgl1 replaced the removed mesa-glx package)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces listens on 7860; the FastAPI backend stays internal on 8000
EXPOSE 7860

# exec keeps streamlit as the signal-receiving foreground process
CMD python app.py & exec streamlit run frontend.py --server.port 7860 --server.address 0.0.0.0
