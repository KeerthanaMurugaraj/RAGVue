
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Pydantic is used for data validation and serialization.
# The use of Pydantic ensures type safety and validation of the data structures.

# Each EvalItem represents a single question-answer pair with optional contexts and metadata.
# MetricResult captures the outcome of a specific metric evaluation.
# ItemEvaluation aggregates the results of multiple metrics for a single EvalItem.
# EvalReport summarizes the evaluations across multiple items.

# This module can be imported and used in various parts of the evaluation framework.
# It promotes code reuse and standardization of evaluation data formats.
# The models can be integrated with evaluation agents, orchestrators, and reporting tools.
# Overall, this module provides a solid foundation for managing evaluation data in a structured manner.

# ----------- Schema Models -----------

class EvalItem(BaseModel):
    question: str
    answer: str
    contexts: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MetricResult(BaseModel):
    name: str
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)

class ItemEvaluation(BaseModel):
    item: EvalItem
    metrics: List[MetricResult]
    aggregate: Optional[float] = None

class EvalReport(BaseModel):
    results: List[ItemEvaluation]
    summary: Dict[str, Any] = Field(default_factory=dict)
