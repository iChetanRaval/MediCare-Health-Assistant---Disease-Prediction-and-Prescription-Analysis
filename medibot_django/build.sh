#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -o errexit

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Apply database migrations
python manage.py migrate

# Start the Django development server
python manage.py runserver 127.0.0.1:8000