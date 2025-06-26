# Stage 1: Use a specific and smaller base image
FROM python:3.9-slim-bullseye

# Set the working directory
WORKDIR /app

# Create a non-root user to run the application for better security
RUN useradd --create-home appuser

# Copy requirements file first to leverage Docker's layer caching
COPY --chown=appuser:appuser requirements.txt .

# Install dependencies as the new user
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-root user
USER appuser

# Copy the rest of the application code
# This will be mounted over by docker-compose for development
COPY --chown=appuser:appuser . .

# Expose the port the app runs on
EXPOSE 3000

# Command to run the application
# Note: The original Dockerfile used port 8000, but the compose file maps 3000.
# The app.py runs on 3000 when executed directly. Sticking to 3000.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
