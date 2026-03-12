import json
from jinja2 import Template

class ThreatQuestionGenerator:

    def __init__(self, llm, template_path):
        self.llm = llm
        with open(template_path, "r") as f:
            self.template = Template(f.read())

    def generate(self, report, **kwargs):
        prompt = self.template.render(report=report, **kwargs)
        response = self.llm.generate(prompt)

        try:
            # Attempt to find JSON block if LLM adds preamble
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                response = "{" + response.split("{", 1)[1].rsplit("}", 1)[0] + "}"

            return json.loads(response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None
