import os
import json
from tqdm import tqdm
import requests

BASE_PATH = ""

with open("prompts/translation_evaluation_prompt_v3.txt", "r", encoding="utf-8") as f:
    evaluation_prompt = f.read()

def read_corpus(dataset_name, file_name):
    english_folder = os.path.join('datasets', dataset_name)
    turkish_folder = os.path.join('translated_datasets', dataset_name)

    corpus = []
    with open(os.path.join(english_folder, f'{file_name}.jsonl'), 'r', encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))

    turkish_corpus = []
    with open(os.path.join(turkish_folder, f'{file_name}.jsonl'), 'r', encoding="utf-8") as f:
        for line in f:
            turkish_corpus.append(json.loads(line))

    if len(corpus) != len(turkish_corpus):
        raise ValueError('File sizes are not equal.')

    return corpus, turkish_corpus

def evaluate(original_text, translated_text):
    data = {
        "model": "aya-expanse:latest",
        "prompt": evaluation_prompt.format(original_text=original_text, translated_text=translated_text),
        "stream": False,
        "required": ["decision"]
    }
    response = requests.post("http://localhost:11434/api/generate", json=data)
    response_text = json.loads(response.text)['response']
    if "pass" in response_text.lower():
        decision = True
    else:
        decision = False
    return decision, response_text

def evaluate_file(original_file, translated_file, save_path):
    responses = []

    with open(save_path, 'r') as f:
        pass_fail_indices = json.load(f)

    for i in tqdm(range(len(original_file))):
        if pass_fail_indices[i] is True:
          responses.append("")
          continue
        eng_text = original_file[i]['text']
        tur_text = translated_file[i]['text']


        decision, response = evaluate(eng_text, tur_text)
        pass_fail_indices[i] = decision
        responses.append(response)

    return pass_fail_indices, responses


def main(dataset_name):
    print(dataset_name)
    corpus, turkish_corpus = read_corpus(dataset_name, 'corpus')
    queries, turkish_queries = read_corpus(dataset_name, 'queries')

    corpus_save_path = os.path.join('llm_evaluations_v2', dataset_name, 'corpus_pass_fail_indices.json')
    queries_save_path = os.path.join('llm_evaluations_v2', dataset_name, 'queries_pass_fail_indices.json')

    corpus_pass_fail_indices, corpus_responses = evaluate_file(corpus, turkish_corpus, corpus_save_path)
    queries_pass_fail_indices, query_responses = evaluate_file(queries, turkish_queries, queries_save_path)

    with open(corpus_save_path, "w") as f:
        json.dump(corpus_pass_fail_indices, f)

    with open(queries_save_path, "w") as f:
        json.dump(queries_pass_fail_indices, f)

    with open(os.path.join('llm_evaluations_v2', dataset_name, 'corpus_responses.json'), 'w') as f:
        json.dump(corpus_responses, f)

    with open(os.path.join('llm_evaluations_v2', dataset_name, 'queries_responses.json'), 'w') as f:
        json.dump(query_responses, f)
    
    
datasets = ["arguana", "cqadupstack/gaming", "fiqa", "nfcorpus", "scidocs", "scifact", "quora"]

for dataset in datasets:
  main(dataset)


