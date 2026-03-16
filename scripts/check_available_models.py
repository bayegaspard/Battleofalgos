import requests
import json

def get_working_models():
    api_key = "AIzaSyDPTQSgRt29_uju90qm1oZBQGRgPPTHp1Q"
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if 'error' in data:
            print(f"API Error: {data['error']}")
            return
            
        models = data.get('models', [])
        print("Models supporting generateContent:")
        for m in models:
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                print(f"- {m['name']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_working_models()
