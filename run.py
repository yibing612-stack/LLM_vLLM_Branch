import os
import re
import json
import time
import datetime
from typing import List, Dict, Any, Tuple

import requests

from inferencePipeline import loadPipeline


def _print_progress(done: int, total: int, prefix: str = "") -> None:
    bar_w = 30
    filled = int(bar_w * done / total) if total else 0
    bar = "#" * filled + "-" * (bar_w - filled)
    print(f"\r{prefix}[{bar}] {done}/{total}", end="", flush=True)


def _read_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def _print_qa_preview(items: List[Dict[str, Any]]) -> None:
    """
    打印每道题的学科、题干、标准答案和模型答案，方便人工检查。
    """
    print("=" * 80)
    print("Preview gold answers vs model answers")
    print("=" * 80)
    for row in items:
        subj = row.get("subject", "unknown")
        qid = row.get("questionID", "")
        q = row.get("question", "").strip()
        gold = row.get("answer", "").strip()
        model = row.get("modelAnswer", "").strip()

        print(f"[{subj}] QID={qid}")
        print(f"Q: {q}")
        print(f"Gold : {gold}")
        print(f"Model: {model}")
        print("-" * 80)




def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ollama_judge(question: str, ref_answer: str, cand_answer: str, model: str, host: str) -> int:
    prompt = (
        "You are a strict evaluator. Given a question, a reference answer, and a candidate answer, "
        "decide if the candidate is semantically correct for the question. Minor wording differences are okay; "
        "factual contradictions or wrong results are incorrect. Respond with a single digit: 1 for correct, 0 for incorrect.\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {ref_answer}\n"
        f"Candidate Answer: {cand_answer}\n\n"
        "Output only 1 or 0."
    )

    url = host.rstrip('/') + '/api/generate'
    try:
        resp = requests.post(url, json={
            'model': model,
            'prompt': prompt,
            'stream': False
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get('response', '').strip()
        m = re.search(r'[01]', text)
        return int(m.group(0)) if m else 0
    except Exception as e:
        print(f"\n[eval] Ollama request failed: {e}")
        return 0


def _evaluate_with_ollama(items: List[Dict[str, Any]], model: str, host: str) -> Tuple[Dict[str, Tuple[int, int]], int]:
    by_subject: Dict[str, Tuple[int, int]] = {}
    correct_total = 0

    total = len(items)
    done = 0
    for row in items:
        q = row.get('question', '')
        ref = row.get('answer', '')
        cand = row.get('modelAnswer', '')
        subj = row.get('subject', 'unknown')

        res = _ollama_judge(q, ref, cand, model=model, host=host)
        correct_total += 1 if res == 1 else 0

        c, t = by_subject.get(subj, (0, 0))
        by_subject[subj] = (c + (1 if res == 1 else 0), t + 1)

        done += 1
        _print_progress(done, total, prefix="Evaluate ")
    print()
    return by_subject, correct_total


if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), os.environ.get("DATASET_PATH", "testset_50.json"))
    results_root = os.path.join(os.path.dirname(__file__), os.environ.get("RESULTS_DIR", "results"))
    skip_evaluation = os.environ.get("SKIP_EVALUATION", "1") == "1"  # Set to "1" to skip Ollama evaluation
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:32b")

    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    rows = _read_dataset(dataset_path)

    # Load pipeline (uses default mode from load.py)
    pipeline = loadPipeline()

    # Prepare questions for inference
    questions = [{"questionID": r["questionID"], "question": r["question"]} for r in rows]

    # Inference: strictly follow documentation style
    start_time = time.perf_counter()
    answers = pipeline(questions)
    infer_elapsed = time.perf_counter() - start_time

    # Map answers by questionID
    id_to_ans = {a['questionID']: a['answer'] for a in answers}

    # Build augmented results (original + modelAnswer)
    augmented: List[Dict[str, Any]] = []
    for r in rows:
        new_r = dict(r)
        new_r['modelAnswer'] = id_to_ans.get(r['questionID'], '')
        augmented.append(new_r)

    _print_qa_preview(augmented)

    # Persist outputs to timestamped directory
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(results_root, ts)
    _ensure_dir(out_dir)

    aug_json_path = os.path.join(out_dir, 'predictions_with_model_answers.json')
    with open(aug_json_path, 'w', encoding='utf-8') as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    total_count = len(augmented)

    # Evaluation via Ollama (progress bar). This time is excluded from inference time.
    if not skip_evaluation:
        eval_start = time.perf_counter()
        by_subject, correct_total = _evaluate_with_ollama(augmented, model=ollama_model, host=ollama_host)
        eval_elapsed = time.perf_counter() - eval_start
        overall_acc = (correct_total / total_count) if total_count else 0.0

    # Prepare metrics text
    lines: List[str] = []
    lines.append(f"Dataset: {os.path.basename(dataset_path)}")
    lines.append(f"Total questions: {total_count}")
    lines.append("")
    
    if not skip_evaluation:
        lines.append("Per-subject accuracy:")
        for subj in sorted(by_subject.keys()):
            c, t = by_subject[subj]
            acc = (c / t) if t else 0.0
            lines.append(f"  - {subj}: {acc*100:.2f}% ({c}/{t})")
        lines.append("")
        lines.append(f"Overall accuracy: {overall_acc*100:.2f}% ({correct_total}/{total_count})")
        lines.append("")
    
    lines.append(f"Inference time (s): {infer_elapsed:.2f}")
    lines.append(f"Average per question (s): {infer_elapsed/total_count:.2f}")
    
    if not skip_evaluation:
        lines.append(f"Evaluation time (s): {eval_elapsed:.2f}")
        lines.append("")
        lines.append(f"Ollama host: {ollama_host}")
        lines.append(f"Ollama model: {ollama_model}")
    else:
        lines.append("")
        lines.append("Evaluation: SKIPPED (inference only mode)")

    metrics_path = os.path.join(out_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # Console summary
    print(f"Saved augmented results to: {aug_json_path}")
    print(f"Saved metrics to: {metrics_path}")
    if not skip_evaluation:
        print(f"Overall accuracy: {overall_acc*100:.2f}% | Inference: {infer_elapsed:.2f}s (avg {infer_elapsed/total_count:.2f}s/q)")
    else:
        print(f"Inference only: {infer_elapsed:.2f}s (avg {infer_elapsed/total_count:.2f}s/q)")
