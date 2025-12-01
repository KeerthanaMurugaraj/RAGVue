from pathlib import Path
from ragvue import EvaluationAgent # from ragvue.agents.orchestrator import EvaluationAgent
from ragvue import ReportBuilder, save_all_formats  # save_all_formats writes md/json/html # from ragvue.reporting.report import ReportBuilder, save_all_formats

METRICS = []
items = [
    {
        "question": "When did the Berlin Wall fall?",
        "answer": "The Berlin Wall fell in 1991 after months of protests.",
        "contexts": [
            "The Berlin Wall effectively fell on November 9, 1989, when border crossings were opened.",
            "Formal reunification of Germany occurred on October 3, 1990.",
        ],
    },
]

agent = EvaluationAgent(metrics=METRICS)
report_model = agent.evaluate_items(items)   # Pydantic EvalReport

report_dict = report_model.model_dump()      # convert for ReportBuilder
md = ReportBuilder(report_dict).to_markdown()

out_dir = Path("runs/calib_faith_smoke")
out_dir.mkdir(parents=True, exist_ok=True)

# Write Markdown yourself
(out_dir / "report.md").write_text(md, encoding="utf-8")

# # Also dump json/html via the helper (keeps filenames consistent)
# save_all_formats(report_dict, out_dir)
#
# print(f"Saved:\n- {out_dir / 'report.md'}\n- {out_dir / 'report.json'}\n- {out_dir / 'report.html'}")


