#!/bin/bash

# Quick start script for testing violations

echo "=========================================="
echo "Violation Test Script - Quick Start"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  Please edit .env file and configure your LLM settings:"
    echo "   - Set API_KEY for OpenAI"
    echo "   - Or configure vLLM/HuggingFace settings"
    echo ""
    echo "Example configurations:"
    echo ""
    echo "For OpenAI:"
    echo "  LLM_BACKEND_TYPE=openai"
    echo "  LLM_MODEL_NAME=gpt-3.5-turbo"
    echo "  API_KEY=your_api_key_here"
    echo ""
    echo "For vLLM (local server):"
    echo "  LLM_BACKEND_TYPE=vllm"
    echo "  BASE_URL=http://localhost:8000/v1"
    echo "  API_KEY=dummy_key"
    echo ""
    exit 1
fi

echo "✅ Found .env file"
echo "Running tests..."
echo ""

# Run the test script
python scripts/test_violations.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "Results saved to: data/test_results.json"
echo "=========================================="
