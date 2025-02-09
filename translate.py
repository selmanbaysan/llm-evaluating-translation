from vllm import LLM, SamplingParams

class OpenSourceTranslation:
    def __init__(self, model_path="CohereForAI/aya-expanse-8b", prompt_path="prompts/translate_prompt.txt"):
        self.model = LLM(model=model_path)
        self.sampling_params = SamplingParams(use_beam_search=True, best_of=3, max_tokens=512, length_penalty=1.2, temperature=0)
        with open(prompt_path, "r") as f:
            self.translate_prompt = f.read()
        
    def translate(self, text):
        input_prompt = self.translate_prompt.format(text=text)
        output = self.model.generate(input_prompt, self.sampling_params)
        return output.text


    