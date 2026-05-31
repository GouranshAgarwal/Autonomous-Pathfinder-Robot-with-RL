FROM python:3.10-slim

WORKDIR /app

# Install minimal build tools required for compiling certain wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 🔥 CRITICAL OPTIMIZATION: Install CPU-only PyTorch first.
# This downloads a lightweight version (~150MB instead of 2.5GB+) and blocks 
# subsequent packages from pulling down heavy CUDA binaries.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest of your stack
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]