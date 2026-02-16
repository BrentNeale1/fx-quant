FROM python:3.11-slim

# LightGBM needs libgomp
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY config/system.yaml config/system.yaml
COPY src/ src/
COPY sql/ sql/

RUN mkdir -p logs models

ENV PYTHONUNBUFFERED=1
CMD ["python", "src/order_executor.py"]
