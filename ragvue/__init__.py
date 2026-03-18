from dotenv import load_dotenv

try:
    load_dotenv(override=False)
except Exception:
    pass

from .src.core.manual_mode import EvaluationAgent, evaluate
from .src.core.agentic_mode import AgenticOrchestrator
from .src.core.metrics_loader import load_metrics
from .src.reporting.report import *
from .src.core.utils import *
from .src.core.base import *
from .src.core.aspects import *

__all__ = [
    "evaluate",
    "EvaluationAgent",
    "AgenticOrchestrator",
    "load_metrics",
    "ReportBuilder",
    "save_all_formats",
    "get_aspects",
    "chat_once",
    "get_openai_client",
]
