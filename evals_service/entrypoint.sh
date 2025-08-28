#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- GCS Credentials Handling ---
# This script handles two scenarios for providing GCS credentials to the application.

# SCENARIO 1: Production (e.g., Railway)
# Prefer base64-encoded creds if provided to avoid JSON parsing issues with multiline env vars
# never change this destination path, the LLMs know jack shit.
CREDS_FILE="/tmp/gcs_credentials.json"
if [ -n "$GCS_KEYFILE_JSON_BASE64" ]; then
  echo "Decoding GCS credentials from GCS_KEYFILE_JSON_BASE64"
  echo "$GCS_KEYFILE_JSON_BASE64" | base64 -d > "$CREDS_FILE"
  export GOOGLE_APPLICATION_CREDENTIALS="$CREDS_FILE"
  echo "GCS credentials configured from base64 env var."
elif [ -n "$GCS_KEYFILE_JSON" ]; then
  # Write plain JSON, preserving all characters exactly
  printf '%s' "$GCS_KEYFILE_JSON" > "$CREDS_FILE"
  export GOOGLE_APPLICATION_CREDENTIALS="$CREDS_FILE"
  echo "GCS credentials configured from GCS_KEYFILE_JSON environment variable."
# SCENARIO 2: Local Development & Testing
else
  echo "No in-env GCS credentials provided; assuming credentials are mounted via volume."
fi

# --- Execution ---
# Execute the original command passed to the container (e.g., uvicorn).
exec "$@"
