
RAGVue: A Diagnostic View for Explainable and Automated Evaluation of Retrieval-Augmented Generation
---

RAGVue is a lightweight and explainable evaluation framework for Retrieval-Augmented Generation (RAG) systems. It provides fine-grained, interpretable diagnostics across retrieval, answer quality, and factual grounding, going beyond simple numerical metrics.

RAGVue supports:

- 🔍 **Manual Mode** – you explicitly choose metrics  
- 🤖 **Agentic Mode** – automatically selects the right metrics  
- 🖥️ **Streamlit UI** – no-code interactive evaluation  
- 🔧 **Multiple Interfaces:**  
  - Python API  
  - Python CLI runner (`ragvue-py`)  
  - CLI tool (`ragvue-cli`)  
  - Streamlit Web UI  

RAGVue includes **seven core metrics** across retrieval, answer quality, and grounding.

## 🚀 Installation

### Install from source

```
git clone <-repo-url> ragvue
cd ragvue
pip install -e .
```
## 🧠 Usage

RAGVue can be used via:

- **A. Python API**  
- **B. CLI tools (`ragvue-cli` & `ragvue-py`)**  
- **C. Streamlit UI (no-code)**  

### A. Python API

```python
from ragvue import evaluate, load_metrics

items = [
    {"question": "...", "answer": "...", "context": [...]}
]

metrics = load_metrics().keys()
report = evaluate(items, metrics=list(metrics))

print(report)
```

### B. Command-Line Interface (CLI)

#### **1. `ragvue-cli` (main CLI)**

#### Help & List available metrics
```
ragvue-cli --help
ragvue-cli list-metrics
```
####  Manual mode
```
ragvue-cli eval   --inputs <your_data.jsonl>   --metrics <metric_name>   --out-base report_manual   --formats "json,md,csv"
```
#### Agentic mode
```
ragvue-cli agentic   --inputs <your_data.jsonl>   --out-base report_agentic --formats "json,md,csv"
```
#### **2. `ragvue-py` (lightweight Python runner)**

#### Help
```
ragvue-py --help
```

#### Manual mode
```
ragvue-py   --input <your_data.jsonl>   --metrics <metrics>   --out-base report_manual   --skip-agentic
```

#### Agentic mode
```
ragvue-py   --input <your_data.jsonl>  --metrics <metrics> --agentic-out report_agentic   --skip-manual
```
### C. Streamlit UI (No-Code Interface)

Launch the UI:
```
streamlit run streamlit_app.py
```

#### Features
- Upload JSONL files  
- Manual & Agentic metric selection  
- API key input  
- Global summary dashboard  
- Individual case-level diagnostic views  
- Multi-format export (JSON, Markdown, CSV, HTML)  

---

### 📄 Input Format

RAGVue expects JSONL like:

```json
{"question": "...", "answer": "...", "context": ["chunk1", "chunk2"]}
```
###  Metrics Overview
| **Category**             | **Metric**            | **Inputs**  |  **Description**                                                     |
|--------------------------|-----------------------|-------------| ---------------------------------------------------------------------- |
| **Retrieval Metrics**    | *Retrieval Relevance* | Q, C        | Evaluates how useful each retrieved chunk is for addressing the information needs of the question, based on per-chunk relevance scoring.          |
|                          | *Retrieval Coverage*  | Q, C        | Assesses whether the retrieved context collectively provides sufficient coverage for all sub-aspects required to answer the question. |
| **Answer Metrics**       | *Answer Relevance*    | Q, A        | Measures how well the answer aligns with the intent and scope of the question, identifying missing, irrelevant, or off-topic content.        |
|                          | *Answer Completeness* | Q, A        | Determines whether the answer fully addresses all aspects of the question without omissions.      |
|                          | *Clarity*             | A           | Evaluates the linguistic quality of the answer, including grammar, fluency, logical flow, coherence, and overall readability.               |
|**Grounding & Stability** | *Strict Faithfulness* | A, C        | Evaluates how many factual claims in the answer are directly supported by the retrieved context, enforcing strict evidence alignment (entity accuracy and temporal correctness)|
|                          | *Calibration*         | Q, A, C     | Examines the stability of metric by measuring variance across different judge configurations (model choice and temperature).         |

### 🔐 Licensing 

This project is currently licensed under **CC BY-NC-ND 4.0**.
Because the framework is still an early prototype, the license is intentionally restrictive.  A more open license will be introduced once the system matures.
This license allows:  
✔ research & academic use  
❌ no commercial use  
❌ no modification or derivative works  
❌ no redistribution of modified versions  
For full license text, see: 
https://creativecommons.org/licenses/by-nc-nd/4.0/

###  📩 Contact
For questions or licensing inquiries, please contact: [ragvue.license@gmail.com](mailto:ragvue.license@gmail.com)

### 📚 Citation

(To be added)


