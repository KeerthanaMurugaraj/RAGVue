from ragvue import evaluate, ReportBuilder

items = [{
    "question":"When was SpaceX founded?",
    # "answer":"Elon Musk founded SpaceX in 2002.",
   "answer":"2002",
    "contexts":["SpaceX was founded by Elon Musk in 2002",
                "The company aims to reduce space transportation costs to enable the colonization of Mars.",
                "Elon Musk is also the CEO of Tesla, Inc.",
                "i like apples"
                 ]
}]

# report = evaluate(items, metrics=["answer_completeness","faithfulness","clarity"])
report = evaluate(items, metrics=[])
print(report)


