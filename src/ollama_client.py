import requests

class OllamaClient:

    def __init__(self, model="gpt-oss:20b", host="http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, prompt):

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]
