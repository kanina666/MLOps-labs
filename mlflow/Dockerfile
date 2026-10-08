FROM python:3.12-slim

ARG MLFLOW_VERSION=2.22.0

# psycopg2-binary: драйвер для Backend Store (PostgreSQL), boto3: клиент для Artifact Store (S3/MinIO)
RUN pip install --no-cache-dir \
    mlflow==${MLFLOW_VERSION} \
    "sqlalchemy>=2.0.38,<2.1" \
    psycopg2-binary==2.9.10 \
    boto3==1.35.99

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
