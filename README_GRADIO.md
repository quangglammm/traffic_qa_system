# Gradio Demo - Traffic QA System

## Overview

Interactive web interface for testing the Traffic Violation Question Answering System.

## Installation

Install Gradio dependency:

```bash
pip install gradio
```

## Running the Demo

### Basic Usage

```bash
python run_gradio_demo.py
```

The app will be available at: `http://localhost:7860`

### Custom Configuration

```python
from src.presentation.gradio_app import create_app

app = create_app()
app.launch(
    server_name="0.0.0.0",
    server_port=8080,
    share=True,  # Create public link
    auth=("username", "password")  # Optional authentication
)
```

## Features

- 💬 **Chat Interface**: Natural conversation flow
- 📝 **Question History**: View previous questions and answers
- 📚 **Legal Citations**: Automatic citation formatting
- 🔄 **Clear History**: Reset conversation anytime
- 📱 **Responsive**: Works on desktop and mobile

## Example Questions

Try asking:
- "Xe máy vượt đèn đỏ bị phạt bao nhiêu?"
- "Ô tô quá tốc độ 20km/h ở nội thành Hà Nội bị phạt thế nào?"
- "Điều luật nào quy định về không đội mũ bảo hiểm?"
- "Hình phạt bổ sung khi say rượu lái xe là gì?"

## Configuration

The Gradio app uses the same dependency injection container as the FastAPI app.
Configure backends in `src/presentation/di_container.py`:

```python
# Configure LLM backend
container.llm_service.override(
    UniversalLLMAdapter(
        backend_type="openai",  # or "huggingface"
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
)

# Configure embedding backend
container.embedding_service.override(
    UniversalEmbeddingAdapter(
        backend_type="openai",  # or "huggingface"
        model_name="text-embedding-ada-002",
        api_key="your-api-key"
    )
)
```

## Troubleshooting

### Port Already in Use

Change the port:
```bash
python run_gradio_demo.py --port 8080
```

Or modify `run_gradio_demo.py`:
```python
app.launch(server_port=8080)
```

### Share Link Not Working

Enable sharing:
```python
app.launch(share=True)
```

This creates a temporary public URL (valid for 72 hours).
