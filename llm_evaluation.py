#from vllm import LLM, SamplingParams
import requests
import json

class Evaluation:
    def __init__(self, model_path="CohereForAI/aya-expanse-8b", prompt_path="prompts/translation_evaluation_prompt_v2.txt"):
        #self.model = LLM(model=model_path)
        #self.sampling_params = SamplingParams(use_beam_search=True, best_of=3, max_tokens=512, length_penalty=1.2, temperature=0)
        with open(prompt_path, "r") as f:
            self.evaluation_prompt = f.read()
        self.url = "http://localhost:11434/api/generate"
    
    def evaluate(self, original_text, translated_text):
        data = {
            "model": "aya-expanse:latest",
            "prompt": self.evaluation_prompt.format(original_text=original_text, translated_text=translated_text),
            "stream": False,
            "required": ["decision"]
        }
        response = requests.post(self.url, json=data)
        return json.loads(response.text)['response']
    