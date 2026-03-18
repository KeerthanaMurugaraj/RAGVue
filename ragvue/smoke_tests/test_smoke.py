from ragvue import evaluate , load_metrics

def main():
    items = [{
        "question": "Who founded SpaceX?",
        "answer": "Elon Musk.",
        "contexts": ["SpaceX was founded by Elon Musk in 2002."]
    }]
    # metrics = list(load_metrics().keys()) # for all metrics
    metrics = ["answer_completeness", "answer_relevance"]
    out = evaluate(items, metrics=metrics)

    print("Report:", out)

if __name__ == "__main__":
    main()
