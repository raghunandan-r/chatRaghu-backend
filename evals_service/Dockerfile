# Stage 1: Use a specific and smaller base image
FROM python:3.11-slim

WORKDIR /app

# Copy everything to app root (revert to original)
COPY . ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Create storage directories
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/audit_data /app/eval_results && \
    chown -R appuser:appuser /app/audit_data /app/eval_results && \
    chmod -R 755 /app/audit_data /app/eval_results

USER appuser

# Copy entrypoint
COPY --chown=appuser:appuser entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["./entrypoint.sh"]

# Expose the port the app runs on
EXPOSE 8001

# Restore original CMD with Railway compatibility
CMD ["sh", "-c", "uvicorn app:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-${PORT:-8001}}"]
