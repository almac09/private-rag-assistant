# Setup script for private-rag-assistant
# Run this to set up the project environment

Write-Host "Setting up private-rag-assistant..." -ForegroundColor Cyan

# Install Ollama if not already installed
Write-Host "Installing Ollama..." -ForegroundColor Yellow
winget install Ollama.Ollama -e

# Pull the llama3 model
Write-Host "Pulling llama3 model..." -ForegroundColor Yellow
ollama pull llama3

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
uv venv .venv

# Activate and sync dependencies
Write-Host "Activating virtual environment and installing dependencies..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"
uv sync

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Run 'python 00_hello_world.py' to test LLM providers."