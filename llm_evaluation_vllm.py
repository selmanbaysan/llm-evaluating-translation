from vllm import LLM, SamplingParams

class Evaluation:
    def __init__(self, model_path="CohereForAI/aya-expanse-8b", prompt_path="prompts/translation_evaluation_prompt_v2.txt"):
        self.model = LLM(model=model_path)
        self.sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=100)
        with open(prompt_path, "r") as f:
            self.evaluation_prompt = f.read()
    
    
    def evaluate(self, original_text, translated_text):
        prompt = self.evaluation_prompt.format(original_text=original_text, translated_text=translated_text)
        output = self.model.generate([prompt], self.sampling_params)
        return output[0].outputs[0].text

if __name__ == "main":
    evaluator = Evaluation()
    print(evaluator.evaluate("Hello, how are you?", "Merhaba, nasılsın?"))
    