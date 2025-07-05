# Stage 1: Use a specific and smaller base image
FROM python:3.11-slim-bullseye

# Set the working directory
WORKDIR /app

# Create a non-root user for security
RUN useradd --create-home appuser

# Copy requirements first for layer caching
COPY --chown=appuser:appuser requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create storage directories with proper permissions
RUN mkdir -p /app/audit_data /app/eval_results && \
    chown -R appuser:appuser /app/audit_data /app/eval_results && \
    chmod -R 755 /app/audit_data /app/eval_results

# Switch to the non-root user
USER appuser

# Copy the application code
# This will be mounted over by docker-compose for development
COPY --chown=appuser:appuser . .

# Copy the entrypoint script and make it executable
COPY --chown=appuser:appuser entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose the port the app runs on
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set the entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# The CMD will be passed to the entrypoint
CMD ["sh", "-c", "uvicorn app:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8001}"]
