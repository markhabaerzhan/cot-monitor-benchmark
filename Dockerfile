FROM python:3.13.16-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: smoke run on the 10-row sample with the offline placeholder monitor
CMD ["python", "-m", "src.main", "--config", "configs/config.yaml"]
