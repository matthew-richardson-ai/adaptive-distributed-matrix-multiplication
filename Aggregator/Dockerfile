FROM python:3.12-slim

WORKDIR /app

COPY Aggregator/requirements.txt /app/Aggregator/requirements.txt
RUN pip install --no-cache-dir -r /app/Aggregator/requirements.txt

COPY Protocol /app/Protocol
COPY Aggregator /app/Aggregator

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "Aggregator.agg_server"]
