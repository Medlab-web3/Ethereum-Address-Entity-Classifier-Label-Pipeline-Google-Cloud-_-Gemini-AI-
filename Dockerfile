FROM python:3.10-slim

WORKDIR /app

COPY . /app

COPY .env /app/.env

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]