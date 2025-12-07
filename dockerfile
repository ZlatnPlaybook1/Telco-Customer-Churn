FROM python:3.11-slim

# Prevent Python stdout buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS dependencies if needed (optional but recommended)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
