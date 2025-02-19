import os
import json

EVALUATION_FOLDER = "evaluation_samples_v2/human_evaluation_samples"

sample_files = os.listdir(EVALUATION_FOLDER)

human_evaluations = []
llm_evaluations = []

for file in sample_files:
    file_path = os.path.join(EVALUATION_FOLDER, file)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data["human_decision"] == "FAIL":
        human_evaluations.append(False)
    else:
        human_evaluations.append(True)

    llm_evaluations.append(data["translation_is_valid"])

# Calculate confusion matrix metrics
true_positives = sum(1 for h, l in zip(human_evaluations, llm_evaluations) if h and l)
true_negatives = sum(1 for h, l in zip(human_evaluations, llm_evaluations) if not h and not l) 
false_positives = sum(1 for h, l in zip(human_evaluations, llm_evaluations) if not h and l)
false_negatives = sum(1 for h, l in zip(human_evaluations, llm_evaluations) if h and not l)

# Calculate exact match
exact_match = sum(1 for h, l in zip(human_evaluations, llm_evaluations) if h == l) / len(human_evaluations)

# Create confusion matrix visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

conf_matrix = np.array([[true_negatives, false_positives],
                        [false_negatives, true_positives]])

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('LLM vs Human Evaluation Confusion Matrix')
plt.xlabel('LLM Prediction')
plt.ylabel('Human Decision')

# Create output directory if it doesn't exist
os.makedirs('llm_alignment_folder', exist_ok=True)

# Save confusion matrix plot
plt.savefig('llm_alignment_folder/confusion_matrix.png')
plt.close()

# Calculate additional metrics
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Save metrics to JSON
metrics = {
    'exact_match': exact_match,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'confusion_matrix': {
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
}
os.makedirs("llm_alignments")
with open('llm_alignments/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

