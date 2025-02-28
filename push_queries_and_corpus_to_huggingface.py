import datasets
import json
import os

from huggingface_hub import upload_file, create_repo, HfApi

api = HfApi()

def read_corpus(dataset_name, file_name):
    turkish_folder = os.path.join('corrected_datasets', dataset_name)

    turkish_corpus = []
    with open(os.path.join(turkish_folder, f'{file_name}.jsonl'), 'r', encoding="utf-8") as f:
        for line in f:
            turkish_corpus.append(json.loads(line))
    
    return turkish_corpus


if __name__ == '__main__':
    dataset_list = ["nfcorpus"] # "fiqa", "scidocs", "quora", "scifact", "arguana", , , "cqadupstack/gaming"
    
    for dataset in dataset_list:
        print(dataset)

        if dataset == "cqadupstack/gaming":
            repo_name = "selmanbaysan/cqadupstack-gaming"
        else:
            repo_name = "selmanbaysan/" + dataset
        
        repo_name += "-tr"

        create_repo(repo_name, exist_ok=True, repo_type="dataset")

        corpus = read_corpus(dataset, 'corpus')
        queries = read_corpus(dataset, 'queries')

        for elem in corpus:
            elem["_id"] = str(elem["_id"])
            elem["title"] = str(elem["title"])
            elem["text"] = str(elem["text"])

        for elem in queries:
            elem["_id"] = str(elem["_id"])
            elem["text"] = str(elem["text"])

        corpus_ds = datasets.Dataset.from_list(corpus)
        queries_ds = datasets.Dataset.from_list(queries)
        qrels = datasets.Dataset.from_csv("corrected_datasets/" + dataset + "/qrels/test.tsv", delimiter="\t")
        qrels.to_json()

        data = {"corpus": corpus_ds, "queries": queries_ds, "default": qrels}

        for key, value in data.items():
            # push to huggingface_hub
            value.push_to_hub(repo_name, key)


        """for splits in ["queries", "corpus"]:
            save_path = f"{splits}.jsonl"
            data[splits].to_json(save_path)
            upload_file(
                path_or_fileobj=save_path,
                path_in_repo=save_path,
                repo_id=repo_name,
                repo_type="dataset",
            )
            os.system(f"rm {save_path}")"""