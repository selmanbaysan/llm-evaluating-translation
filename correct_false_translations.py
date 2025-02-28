import os
import json
from tqdm import tqdm

from translate_with_gemini import GeminiTranslation

translator = GeminiTranslation()

def read_corpus(dataset_name, file_name):
    english_folder = os.path.join('datasets', dataset_name)
    turkish_folder = os.path.join('corrected_datasets', dataset_name)

    corpus = []
    #check corpus sizes are equal
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

def read_evaluations(dataset_name, file_name):

    evaluation_folder = os.path.join("llm_evaluations", dataset_name)
    
    with open(os.path.join(evaluation_folder, f"{file_name}_pass_fail_indices.json"), "r") as f:
        pass_fail_indices = json.load(f)
    
    return pass_fail_indices

def concat_evaluations(dataset, translated_dataset, evaluations):
    translation_sets = []
    for i in range(len(evaluations)):
        translation_sets.append({'english_text': dataset[i]["text"], "turkish_text": translated_dataset[i]["text"], "translation_is_valid": evaluations[i]})

    return translation_sets

def read_error_indices(dataset_name):
    file_path = os.path.join("error_indexes", dataset_name, "error_indexes.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    
    corpus_errors, query_errors = data["corpus_error_indexes"], data["query_error_indexes"]

    return corpus_errors, query_errors

def correct_translation_errors(dataset_name):
    corpus, turkish_corpus = read_corpus(dataset_name, 'corpus')
    queries, turkish_queries = read_corpus(dataset_name, 'queries')

    corpus_error_indexes, query_error_indexes = read_error_indices(dataset_name)

    for idx in corpus_error_indexes:
        print(idx)
        status_code, translation = translator.translate(corpus[idx]["text"])
        try:
            turkish_corpus[idx]["text"] = translation["candidates"][0]["content"]["parts"][0]["text"]
            corpus_error_indexes.remove(idx)
        except:
            print(status_code)
            print(f"Error occured while translating the corpus index {idx}")
            print(corpus[idx])

    for idx in query_error_indexes:
        print(idx)
        status_code, translation = translator.translate(queries[idx]["text"])
        if status_code == 200:
            translation = translation["candidates"][0]["content"]["parts"][0]["text"]
            turkish_queries[idx]["text"] = translation
            query_error_indexes.remove(idx)
        else:
            print(f"Error occured while translating the query index {idx}")
            print(queries[idx])

    return turkish_corpus, turkish_queries, corpus_error_indexes, query_error_indexes


if __name__ == '__main__':
    output_dir = "corrected_datasets"
    error_dir = "error_indexes"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    datasets = ["quora"] #"arguana", "cqadupstack/gaming", "nfcorpus", "scifact", "fiqa", "scidocs"
    
    for dataset in datasets:
        print(dataset)
        output_folder = os.path.join(output_dir, dataset)
        error_folder = os.path.join(error_dir, dataset)
        
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(error_folder, exist_ok=True)

        fixed_corpus, fixed_queries, corpus_error_indexes, query_error_indexes = correct_translation_errors(dataset)

        # Save corrected translations
        with open(os.path.join(output_folder, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for item in fixed_corpus:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(os.path.join(output_folder, "queries.jsonl"), "w", encoding="utf-8") as f:
            for item in fixed_queries:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Save error indexes
        error_data = {
            "corpus_error_indexes": corpus_error_indexes,
            "query_error_indexes": query_error_indexes
        }
        with open(os.path.join(error_folder, "error_indexes.json"), "w") as f:
            json.dump(error_data, f, indent=4)
        