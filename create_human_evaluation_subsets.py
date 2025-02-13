import os
import json
import argparse

import pandas as pd

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

def create_sample(dataset_name):
    corpus, turkish_corpus = read_corpus(dataset_name, 'corpus')
    queries, turkish_queries = read_corpus(dataset_name, 'queries')

    corpus_evaluations = read_evaluations(dataset_name, "corpus")
    query_evaluations = read_evaluations(dataset_name, "queries")

    concatted_corpus_evaluations = concat_evaluations(corpus, turkish_corpus, corpus_evaluations)
    concatted_query_evaluations = concat_evaluations(queries, turkish_queries, query_evaluations)

    corpus_evaluations_df = pd.DataFrame(concatted_corpus_evaluations)
    query_evaluations_df = pd.DataFrame(concatted_query_evaluations)

    true_corpus_evaluations = corpus_evaluations_df[corpus_evaluations_df["translation_is_valid"] == True]
    false_corpus_evaluations =corpus_evaluations_df[corpus_evaluations_df["translation_is_valid"] == False]

    a = true_corpus_evaluations.sample(5, replace=True)
    b = false_corpus_evaluations.sample(5, replace=True)

    true_query_evaluations = query_evaluations_df[query_evaluations_df["translation_is_valid"] == True]
    false_query_evaluations =query_evaluations_df[query_evaluations_df["translation_is_valid"] == False]
    
    c = true_query_evaluations.sample(5, replace=True)
    d = false_query_evaluations.sample(5, replace=True)

    sample_df = pd.concat([a, b, c, d])

    return sample_df

if __name__ == '__main__':
    OUTPUT_PATH = "evaluation_samples"
    datasets = ["arguana", "cqadupstack/gaming", "fiqa", "nfcorpus", "scidocs", "scifact"]
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for dataset in datasets:
        sample_df = create_sample(dataset)
        if dataset == "cqadupstack/gaming":
            dataset = "cqadupstack_gaming"

        sample_df.to_csv(f"{OUTPUT_PATH}/{dataset}.csv")
    



    
