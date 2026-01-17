#!/bin/bash

# VTA Tracking Run Script
# This script sets up and runs the tap detection system

echo "Starting VTA Tracking System..."

# Create necessary directories
mkdir -p logs uploads outputs static

# # Check if virtual environment exists
# if [ ! -d "venv" ]; then
#     echo "Creating virtual environment..."
#     python3 -m venv venv
# fi

# # Activate virtual environment
# echo "Activating virtual environment..."
# source venv/bin/activate

# Install/upgrade dependencies
# echo "Installing dependencies..."
# pip install --upgrade pip
# pip install -r requirements.txt

# Run the application
echo "Starting backend on port 8501..."

# Start backend (serves both API and static files)
uvicorn main:app --host 0.0.0.0 --port 8501

echo "Server running on http://localhost:8501"
echo "Web Interface: http://localhost:8501"
echo "API Documentation: http://localhost:8501/docs"
echo "Press Ctrl+C to stop the server..."
