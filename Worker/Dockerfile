FROM python:3.12-slim

WORKDIR /app

COPY Worker/requirements.txt /app/Worker/requirements.txt
RUN pip install --no-cache-dir -r /app/Worker/requirements.txt

COPY Protocol /app/Protocol
COPY Worker /app/Worker

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "Worker.worker_server"]
