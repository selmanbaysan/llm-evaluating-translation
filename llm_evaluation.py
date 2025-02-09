from vllm import LLM, SamplingParams

class Translation:
    def __init__(self, model_path="CohereForAI/aya-expanse-8b", prompt_path="prompts/translation_evaluation_prompt.txt"):
        self.model = LLM(model=model_path)
        self.sampling_params = SamplingParams(use_beam_search=True, best_of=3, max_tokens=512, length_penalty=1.2, temperature=0)
        self.evaluation_prompt = self.read_prompt(prompt_path)

    def read_prompt(self, prompt_path):
        with open(prompt_path, "r") as f:
            self.translate_prompt = f.read()
        
    def translate(self, text):
        input_prompt = self.translate_prompt.format(text)
        output = self.model.generate(input_prompt, self.sampling_params)
        return output.text


if __name__ == "__main__":
    translator = Translation()
    print(translator.translate("Hello, how are you?"))


    