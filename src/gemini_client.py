import google.generativeai as genai

class GeminiClient:

    def __init__(self, api_key, model="gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt):

        response = self.model.generate_content(prompt)

        return response.text
