# Post process our RAGAS Results, we split it by refusal vs real answer
import csv

REFUSAL_MARKER = "cannot answer" #Our fixed quote to identify refusals (Temperature = 0)

def load_results(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in answer.lower()


def avg(rows: list[dict], metric: str) -> float | None:
    if not rows:
        return None
    return sum(float(r[metric]) for r in rows) / len(rows)


def report(path: str, label: str) -> None:
    results = load_results(path)
    refusals = [r for r in results if is_refusal(r["answer"])]
    substantive = [r for r in results if not is_refusal(r["answer"])]

    print(f"\n=== {label} ({path}) ===")
    print(f"Total: {len(results)} | Refusals: {len(refusals)} | Sustantives: {len(substantive)}")

    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        overall = avg(results, metric)
        ref_avg = avg(refusals, metric)
        sub_avg = avg(substantive, metric)
        ref_str = f"{ref_avg:.3f}" if ref_avg is not None else "n/a"
        sub_str = f"{sub_avg:.3f}" if sub_avg is not None else "n/a"
        print(f"  {metric}: overall={overall:.3f} | refusals={ref_str} | sustantives={sub_str}")


if __name__ == "__main__":
    report("eval/ragas_results_baseline.csv", "Baseline (sin reranking)")
    report("eval/ragas_results_reranked.csv", "Reranked")