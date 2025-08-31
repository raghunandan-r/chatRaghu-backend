# Stage 1: Use a specific and smaller base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Create a non-root user to run the application for better security
RUN useradd --create-home appuser

# Copy requirements file first to leverage Docker's layer caching
COPY --chown=appuser:appuser requirements.txt .

# Install dependencies as the new user
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# Switch to the non-root user
USER appuser

# Copy the rest of the application code
# This will be mounted over by docker-compose for development
COPY --chown=appuser:appuser . .

# Expose the port the app runs on
EXPOSE 3000

# Command to run the application (use env vars for Railway/IPv6 compatibility)
CMD ["sh", "-c", "uvicorn app:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-3000}"]
