#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- GCS Credentials Handling ---
# This script handles two scenarios for providing GCS credentials to the application.

# SCENARIO 1: Production (e.g., Railway)
# The credentials are provided as a multi-line environment variable GCS_KEYFILE_JSON.
if [ -n "$GCS_KEYFILE_JSON" ]; then
  # Create the file that GOOGLE_APPLICATION_CREDENTIALS will point to.
  echo "$GCS_KEYFILE_JSON" > /app/gcs_credentials.json
  # Set the environment variable for the application to use this newly created file.
  export GOOGLE_APPLICATION_CREDENTIALS="/app/gcs_credentials.json"
  echo "GCS credentials configured from GCS_KEYFILE_JSON environment variable."

# SCENARIO 2: Local Development & Testing
# The credentials file is mounted via a Docker volume. GOOGLE_APPLICATION_CREDENTIALS
# is already set in docker-compose.test.yml to point to the mounted file.
else
  echo "GCS_KEYFILE_JSON not set. Assuming credentials are provided via volume mount."
fi

# --- Execution ---
# Execute the original command passed to the container (e.g., uvicorn).
# The application will now find the credentials using the correct method.
exec "$@"
