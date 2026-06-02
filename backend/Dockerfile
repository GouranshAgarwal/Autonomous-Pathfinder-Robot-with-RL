# Use an official lightweight Python image
FROM python:3.10-slim

# Set system environment variables to optimize Python inside Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies if required (gcc for some numpy compilations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's caching layer
COPY requirements.txt /app/

# 🔥 CPU-ONLY OPTIMIZATION: 
# Force install the pure CPU version of torch directly from the PyTorch index.
# This intercept prevents downstream packages from pulling down massive NVIDIA/CUDA wheels.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the port your FastAPI/Uvicorn server runs on
EXPOSE 8000

# Run the WebSocket server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]