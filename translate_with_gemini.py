import requests

class GeminiTranslation:
    def __init__(self, key_path="api_key.txt", prompt_path="prompts/translate_prompt.txt"):
        self.headers = {'Content-Type': 'application/json'}
        self.params = {
            'key': f"{self.read_api_key(key_path)}",
        }
        with open(prompt_path, "r") as f:
            self.translate_prompt = f.read()


    def read_api_key(self, key_path):
        with open(key_path, "r") as f:
            return f.read().strip()


    def create_json_data(self, text):
        json_data = {
            'contents': [
                {
                    'parts': [
                        {
                            'text': self.translate_prompt.format(text=text),
                        },
                    ],
                },
            ],
        }
        return json_data

    def translate(self, text):
        json_data = self.create_json_data(text)
        response = requests.post(
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent',
            params=self.params,
            headers=self.headers,
            json=json_data,
        )
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

