FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY distributedformer/ ./distributedformer/

RUN pip install --no-cache-dir .

RUN mkdir -p logs reports

ENV PYTHONUNBUFFERED=1 \
    DF_DEPTH=2 \
    KV_CAPACITY=100000

# 长驻监控服务 (可用 -e DF_TICKERS / DF_INTERVAL / DF_REALTIME 覆盖)
ENV DF_TICKERS="AAPL TSLA NVDA" \
    DF_INTERVAL=300

CMD ["sh", "-c", "dformer serve --tickers $DF_TICKERS --interval $DF_INTERVAL ${DF_REALTIME:+--realtime}"]
