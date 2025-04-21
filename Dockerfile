# model-service/Dockerfile
# Use Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p configs models scripts

# Copy the entire project
COPY . .

# Make sure the scripts are executable
RUN chmod +x service-setup.sh scripts/setup_role.sh

# Expose the port
EXPOSE 8000

# Command to run the API
CMD ["python", "run.py"]