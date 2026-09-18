FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY budget_agent/ ./budget_agent/
COPY scripts/ ./scripts/
COPY docker/ ./docker/
COPY app.py ./

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["./docker/entrypoint.sh"]
