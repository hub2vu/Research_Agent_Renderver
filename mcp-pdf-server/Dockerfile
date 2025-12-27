FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# Create directories for PDF input and output
RUN mkdir -p /data/pdf /data/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PDF_DIR=/data/pdf
ENV OUTPUT_DIR=/data/output

# Run MCP server
CMD ["python", "server.py"]
