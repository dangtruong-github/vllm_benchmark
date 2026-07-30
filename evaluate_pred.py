import argparse
import json
import os

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_predictions(path):
    predictions = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            predictions[obj["request_id"]] = obj["prediction"].strip()

    return predictions


def exact_match(a, b):
    return float(a == b)


def cosine_similarity_matrix(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/predictions-baseline.jsonl")
    parser.add_argument(
        "--submission", required=True, 
        help="Folder contains result.jsonl for evaluation"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    args = parser.parse_args()

    baseline = load_predictions(args.baseline)
    submission = load_predictions(os.path.join(args.submission, "result.jsonl"))

    common = sorted(set(baseline) & set(submission))

    print(f"Comparing {len(common)} samples")

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    baseline_texts = []
    submission_texts = []

    em_scores = []
    rouge_f = []
    rouge_p = []
    rouge_r = []

    baseline_lengths = []
    submission_lengths = []
    length_ratios = []

    for idx in tqdm(common):
        a = baseline[idx]
        b = submission[idx]

        baseline_texts.append(a)
        submission_texts.append(b)

        em_scores.append(exact_match(a, b))

        r = rouge.score(a, b)["rougeL"]

        rouge_f.append(r.fmeasure)
        rouge_p.append(r.precision)
        rouge_r.append(r.recall)

        la = len(a.split())
        lb = len(b.split())

        baseline_lengths.append(la)
        submission_lengths.append(lb)

        if la == 0:
            length_ratios.append(0.0)
        else:
            length_ratios.append(lb / la)

    print("Loading embedding model...")
    model = SentenceTransformer(args.embedding_model)

    emb_a = model.encode(
        baseline_texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    emb_b = model.encode(
        submission_texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    cosine = cosine_similarity_matrix(emb_a, emb_b)

    EVALUATION_SUMMARY_FILE = os.path.join(args.submission, "evaluation_summary.txt")

    summary_lines = [
        "================ RESULT ================",
        "",
        f"Samples                  : {len(common)}",
        f"Exact Match              : {np.mean(em_scores):.4f}",
        "",
        f"ROUGE-L F1               : {np.mean(rouge_f):.4f}",
        f"ROUGE-L Precision        : {np.mean(rouge_p):.4f}",
        f"ROUGE-L Recall           : {np.mean(rouge_r):.4f}",
        "",
        f"Embedding Cosine         : {np.mean(cosine):.4f}",
        "",
        f"Baseline Avg Length      : {np.mean(baseline_lengths):.2f} words",
        f"Submission Avg Length    : {np.mean(submission_lengths):.2f} words",
        f"Average Length Ratio     : {np.mean(length_ratios):.4f}",
    ]

    summary = "\n".join(summary_lines)

    # Print to console
    print(f"\n{summary}")

    # Save to file
    with open(EVALUATION_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

if __name__ == "__main__":
    main()