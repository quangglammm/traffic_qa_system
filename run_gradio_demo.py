"""
Launch script for Gradio demo.

Usage:
    python run_gradio_demo.py
"""
import sys
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.presentation.gradio_app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Gradio demo"""
    try:
        logger.info("Starting Gradio demo...")
        
        # Create and launch app
        app = create_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start Gradio demo: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
