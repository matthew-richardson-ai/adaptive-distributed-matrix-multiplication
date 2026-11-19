FROM python:3.12-slim

WORKDIR /app

COPY Ingress/requirements.txt /app/Ingress/requirements.txt
RUN pip install --no-cache-dir -r /app/Ingress/requirements.txt

COPY Protocol /app/Protocol
COPY Benchmarks /app/Benchmarks
COPY Ingress /app/Ingress

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "Ingress.ingress_server"]
