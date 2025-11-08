from abc import ABC, abstractmethod

class BaseLLMBackend(ABC):
    """Base class for LLM backends"""
    
    @abstractmethod
    def generate_text(self, prompt: str, max_length: int = 1024) -> str:
        """Generate text using the backend"""
        pass
