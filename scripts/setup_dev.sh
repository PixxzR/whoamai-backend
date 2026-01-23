#!/bin/bash
# Setup development environment for FaceSense API

set -e

echo "Setting up FaceSense API dev environment..."

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision
pip install facenet-pytorch
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create .env from example
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
fi

# Create models directories
mkdir -p models/{specialized,multitask,transfer}
echo "Created models directories"

# Create logs directory
mkdir -p logs
echo "Created logs directory"

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "Pre-commit hooks installed"
fi

echo ""
echo "Dev environment ready!"
echo "Activate venv: source venv/bin/activate"
echo "Run server:    uvicorn app.main:app --reload"
echo "Run tests:     pytest"
