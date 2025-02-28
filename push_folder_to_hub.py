from huggingface_hub import create_repo, HfApi, upload_file
from datasets import Dataset
import json
import os
import pandas as pd

api = HfApi()

def read_corpus(dataset_name, file_name):
    turkish_folder = os.path.join('corrected_datasets', dataset_name)

    turkish_corpus = []
    with open(os.path.join(turkish_folder, f'{file_name}.jsonl'), 'r', encoding="utf-8") as f:
        for line in f:
            turkish_corpus.append(json.loads(line))
    return turkish_corpus

def read_translated_corpus(dataset_name, file_name):
    turkish_folder = os.path.join('translated_datasets', dataset_name)

    turkish_corpus = []
    with open(os.path.join(turkish_folder, f'{file_name}.jsonl'), 'r', encoding="utf-8") as f:
        for line in f:
            turkish_corpus.append(json.loads(line))
    return turkish_corpus

def read_qrels(dataset_name):
    qrels_folder = os.path.join('corrected_datasets', dataset_name, "qrels")

    splits = os.listdir(qrels_folder)
    qrels = {}
    for data_split in splits:
        df = pd.read_csv(os.path.join(qrels_folder, data_split), sep="\t")
    
        qrels[data_split.split(".")[0]] = Dataset.from_pandas(df)
    return qrels

def correct_queries(queries):
    for query in queries:
        query_text = query["text"]
        if "\n\n" in query_text:
            query["text"] = query_text.split("\n\n")[0]
        
    return queries

def correct_data(data, translated_data):
    for i in range(len(data)):
        row = data[i]
        translated_row = translated_data[i]
        if type(row["text"]) == dict:
            try:
                row["text"] = row["text"]["candidates"][0]["content"]["parts"][0]["text"]
            except:
                row["text"] = translated_row["text"]
    return data


def create_data_dict(dataset_name):
    corpus = read_corpus(dataset_name, 'corpus')
    queries = read_corpus(dataset_name, 'queries')

    translated_corpus = read_translated_corpus(dataset_name, 'corpus')
    translated_queries = read_translated_corpus(dataset_name, 'queries')

    corpus = correct_data(corpus, translated_corpus)
    queries = correct_data(queries, translated_queries)

    if dataset_name not in ["arguana"]:
        queries = correct_queries(queries)

    corpus_ds = Dataset.from_list(corpus)
    queries_ds = Dataset.from_list(queries)    
    
    qrels = read_qrels(dataset_name)

    return {
        "corpus": corpus_ds,
        "queries": queries_ds,
        "default": qrels
    }

def upload_data_dict(data_dict, repo_name):
    
    for splits in ["corpus", "queries"]:
        save_path = f"{splits}.jsonl"
        data_dict[splits].to_json(save_path)
        upload_file(
            path_or_fileobj=save_path,
            path_in_repo=save_path,
            repo_id=repo_name,
            repo_type="dataset",
        )
        os.system(f"rm {save_path}")
    
    for split in data_dict["default"]:
        save_path = f"{split}.jsonl"
        data_dict["default"][split].to_json(save_path)
        upload_file(
            path_or_fileobj=save_path,
            path_in_repo=f"qrels/{save_path}",
            repo_id=repo_name,
            repo_type="dataset",
        )
        os.system(f"rm {save_path}")


if __name__ == '__main__':
    dataset_list = ["fiqa", "scidocs", "quora", "scifact", "arguana", "cqadupstack/gaming"] # "nfcorpus"
    
    for dataset in dataset_list:
        print(dataset)

        data_dict = create_data_dict(dataset)

        if dataset == "cqadupstack/gaming":
            repo_name = "selmanbaysan/cqadupstack-gaming"
        else:
            repo_name = "selmanbaysan/" + dataset
        
        repo_name += "-tr"

        create_repo(repo_name, exist_ok=True, repo_type="dataset")
        upload_data_dict(data_dict, repo_name)
        print("Uploaded")
        

        

