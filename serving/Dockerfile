FROM python:3.10-slim

WORKDIR /app/serving

# Install dependencies first for better layer caching
COPY serving/requirements.txt /app/serving/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# The actual source code will be mounted by docker-compose to `/app`
# `app.py` appends `/app` to sys.path so the `preprocessing` module can be imported
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
