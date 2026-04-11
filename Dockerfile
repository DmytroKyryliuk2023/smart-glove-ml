FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/main.py .

ENV FASTAPI_HOST=0.0.0.0
ENV FASTAPI_PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host $FASTAPI_HOST --port $FASTAPI_PORT"]