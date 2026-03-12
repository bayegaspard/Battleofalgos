import yaml
import os
from src.ollama_client import OllamaClient
from src.gemini_client import GeminiClient

class LLMRouter:

    def __init__(self, config_path="config.yaml"):
        # Ensure path is relative to the project root if needed
        # For simplicity, we assume it's in the CWD or provided as an absolute path
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        provider = config["model_provider"]

        if provider == "ollama":
            self.client = OllamaClient(**config["ollama"])

        elif provider == "gemini":
            self.client = GeminiClient(**config["gemini"])
            
        elif provider == "mock":
            from src.mock_client import MockClient
            self.client = MockClient()
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def generate(self, prompt):
        return self.client.generate(prompt)
