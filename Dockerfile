# Small, reproducible image for the model service.
FROM python:3.11-slim

WORKDIR /app

# Install deps FIRST, in their own layer, so code changes don't bust the cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the app code and the versioned model artifacts.
COPY app ./app
COPY models ./models

EXPOSE 8000

# Run the ASGI app with uvicorn; bind to all interfaces inside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
