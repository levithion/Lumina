FROM python:3.11-slim

WORKDIR /app

# Install image processing dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces listens on 7860
EXPOSE 7860

# Start FastAPI in background and Streamlit in foreground
CMD python app.py & streamlit run frontend.py --server.port 7860 --server.address 0.0.0.0
