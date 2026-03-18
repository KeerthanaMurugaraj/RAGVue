
from __future__ import annotations
from typing import Dict, Any
from . import calibration_generic as generic

def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    return generic.evaluate_with_target(item, generic.TARGET_ANSWER_COMPLETENESS)

IS_METRIC = True
