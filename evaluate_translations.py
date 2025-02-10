import argparse
import os
import json
from translate_with_gemini import GeminiTranslation
#from translate import OpenSourceTranslation
from llm_evaluation import Evaluation
import time

gemini_translator = GeminiTranslation()
#open_source_translator = OpenSourceTranslation()
evaluator = Evaluation()

def read_corpus(dataset_name, file_name):
    english_folder = os.path.join('datasets', dataset_name)
    turkish_folder = os.path.join('translated_datasets', dataset_name)

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


def translate_text(text):
    print("Translation made using Gemini translation api")
    return gemini_translator.translate(text)


def evaluate_translation(eng_text, tur_text) -> bool:
    response = evaluator.evaluate(eng_text, tur_text)
    if "PASS" in response:
        return True 
    return False


def evaluate_file(file, translated_file, max_retries=3):
    pass_fail_indices = []
    for i in range(len(file)):
        
        eng_text = file[i]['text']
        tur_text = translated_file[i]['text']
        
        translation_quality = evaluate_translation(eng_text, tur_text)
        
        retry_count = 0
        while translation_quality is False and retry_count < max_retries:
            tur_text = translate_text(eng_text)
            translation_quality = evaluate_translation(eng_text, tur_text)
            retry_count += 1
        
        if retry_count == max_retries:
            print(f'Could not translate the index: {i}')
            pass_fail_indices.append(False)
        else:
            translated_file[i]['text'] = tur_text
            pass_fail_indices.append(True)
        
    return pass_fail_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate translations.')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset that will be evaluated.')
    args = parser.parse_args()

    corpus, turkish_corpus = read_corpus(args.dataset_name, 'corpus')
    queries, turkish_queries = read_corpus(args.dataset_name, 'queries')

    print(len(corpus))
    print(len(queries))

    # get the first 100 of corpus for test purposes
    corpus = corpus[:100]
    turkish_corpus = turkish_corpus[:100]

    start_time = time.time()
    pass_fail_indices = evaluate_file(corpus, turkish_corpus)
    end_time = time.time()

    total_time = end_time - start_time

    print(f"Evaluations take total {total_time} second")

    out_list = []
    for i in range(len(pass_fail_indices)):
        out_list.append({"eng_text":corpus[i], "tur_text":turkish_corpus[i], "decision":pass_fail_indices[i]})
