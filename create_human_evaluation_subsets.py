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

    evaluation_folder = os.path.join("llm_evaluations_v2", dataset_name)
    
    with open(os.path.join(evaluation_folder, f"{file_name}_pass_fail_indices.json"), "r") as f:
        pass_fail_indices = json.load(f)
    
    return pass_fail_indices

def concat_evaluations(dataset, translated_dataset, evaluations):
    translation_sets = []
    for i in range(len(evaluations)):
        translation_sets.append({'english_text': dataset[i]["text"], "turkish_text": translated_dataset[i]["text"], "translation_is_valid": evaluations[i]})

    return translation_sets


def create_sample_df(true_corpus_evaluations, false_corpus_evaluations, true_query_evaluations, false_query_evaluations):
    sample_df = pd.DataFrame(columns=["english_text", "turkish_text", "translation_is_valid"])

    sample_df = pd.concat([sample_df, true_corpus_evaluations.sample(n=5)], ignore_index=True)
    sample_df = pd.concat([sample_df, true_query_evaluations.sample(n=5)], ignore_index=True)

    sample_stats = {'true_corpus_evaluations': 5,
                    'true_query_evaluations': 5
                    }

    # check the number of false evaluations
    false_corpus_count = len(false_corpus_evaluations)
    false_query_count = len(false_query_evaluations)

    if false_corpus_count < 5:
        sample_df = pd.concat([sample_df, false_corpus_evaluations], ignore_index=True)
        sample_stats['false_corpus_evaluations'] = false_corpus_count
    else:
        sample_df = pd.concat([sample_df, false_corpus_evaluations.sample(n=5)], ignore_index=True)
        sample_stats['false_corpus_evaluations'] = 5

    if false_query_count < 5:
        sample_df = pd.concat([sample_df, false_query_evaluations], ignore_index=True)
        sample_stats['false_query_evaluations'] = false_query_count
    else:
        sample_df = pd.concat([sample_df, false_query_evaluations.sample(n=5)], ignore_index=True)
        sample_stats['false_query_evaluations'] = 5

    return sample_df, sample_stats


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
    false_corpus_evaluations = corpus_evaluations_df[corpus_evaluations_df["translation_is_valid"] == False]

    true_query_evaluations = query_evaluations_df[query_evaluations_df["translation_is_valid"] == True]
    false_query_evaluations =query_evaluations_df[query_evaluations_df["translation_is_valid"] == False]

    sample_df, sample_stats = create_sample_df(true_corpus_evaluations, false_corpus_evaluations, true_query_evaluations, false_query_evaluations)
    sample_stats["total_corpus_errors"] = len(false_corpus_evaluations)
    sample_stats["total_query_errors"] = len(false_query_evaluations)

    return sample_df, sample_stats

if __name__ == '__main__':
    OUTPUT_PATH = "evaluation_samples_v2"
    datasets = ["arguana", "cqadupstack/gaming", "fiqa", "nfcorpus", "scidocs", "scifact", "quora"]
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_dir = os.path.join(OUTPUT_PATH, 'human_evaluation_samples')
    os.makedirs(output_dir, exist_ok=True)

    df_list = []
    for dataset in datasets:
        sample_df, sample_stats = create_sample(dataset)
        df_list.append(sample_df)
        if dataset == "cqadupstack/gaming":
            dataset = "cqadupstack_gaming"

        sample_df.to_csv(f"{OUTPUT_PATH}/{dataset}.csv")
        with open(f"{OUTPUT_PATH}/{dataset}_sample_stats.json", 'w') as f:
            json.dump(sample_stats, f)

    combined_df = pd.concat(df_list, ignore_index=True)

    for idx, row in combined_df.iterrows():
        with open(f"{output_dir}/{idx}.json", "w", encoding="utf-8") as f:
            json.dump(row.to_dict(), f, indent=4, ensure_ascii=False)    
