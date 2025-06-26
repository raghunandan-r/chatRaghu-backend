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

# Switch to the non-root user
USER appuser

# Copy the application code
# This will be mounted over by docker-compose for development
COPY --chown=appuser:appuser . .

# Expose the port the app runs on
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
