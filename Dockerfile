FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV PYTHONUNBUFFERED=1
ENV PINECONE_API_KEY=${PINECONE_API_KEY}

CMD ["uvicorn", "app:application", "--host", "0.0.0.0", "--port", "8080"] 