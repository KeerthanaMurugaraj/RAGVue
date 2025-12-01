from __future__ import annotations
from typing import Dict, Any
import os, importlib

# ----------------- TARGET DEFINITIONS -----------------

TARGET_STRICT_FAITHFULNESS = {
    "metric_name": "strict_faithfulness",
    "module_path": "ragvue.src.metrics.strict_faithfulness",
    "model_env": "FAITHFULNESS_MODEL",
    "temp_env": "FAITHFULNESS_TEMPERATURE",
    "judges": [
        {"name": "sf_gpt4omini_t00_run1", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "sf_gpt4omini_t00_run2", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "sf_gpt4omini_t20",      "model": "gpt-4o-mini",   "temperature": 0.2},
        {"name": "sf_gpt4omini_t50",      "model": "gpt-4o-mini",   "temperature": 0.5},
        {"name": "sf_gpt41mini_t00",      "model": "gpt-4.1-mini",  "temperature": 0.0},
        {"name": "sf_gpt41mini_t30",      "model": "gpt-4.1-mini",  "temperature": 0.3},

    ],
}

TARGET_ANSWER_COMPLETENESS = {
    "metric_name": "answer_completeness",
    "module_path": "ragvue.src.metrics.answer_completeness",
    "model_env": "ANSWER_COMPLETENESS_MODEL",
    "temp_env": "ANSWER_COMPLETENESS_TEMPERATURE",
    "judges": [
        {"name": "ac_gpt4omini_t00_run1", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "ac_gpt4omini_t00_run2", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "ac_gpt4omini_t20",      "model": "gpt-4o-mini",   "temperature": 0.2},
        {"name": "ac_gpt4omini_t50",      "model": "gpt-4o-mini",   "temperature": 0.5},
        {"name": "ac_gpt41mini_t00",      "model": "gpt-4.1-mini",  "temperature": 0.0},
        {"name": "ac_gpt41mini_t30",      "model": "gpt-4.1-mini",  "temperature": 0.3},

    ],
}

TARGET_ANSWER_RELEVANCE = {
    "metric_name": "answer_relevance",
    "module_path": "ragvue.src.metrics.answer_relevance",
    "model_env": "ANSWER_RELEVANCE_MODEL",
    "temp_env": "ANSWER_RELEVANCE_TEMPERATURE",
    "judges": [
        {"name": "ar_gpt4omini_t00_run1", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "ar_gpt4omini_t00_run2", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "ar_gpt4omini_t20",      "model": "gpt-4o-mini",   "temperature": 0.2},
        {"name": "ar_gpt4omini_t50",      "model": "gpt-4o-mini",   "temperature": 0.5},
        {"name": "ar_gpt41mini_t00",      "model": "gpt-4.1-mini",  "temperature": 0.0},
        {"name": "ar_gpt41mini_t30",      "model": "gpt-4.1-mini",  "temperature": 0.3},

    ],
}

TARGET_CLARITY = {
    "metric_name": "clarity",
    "module_path": "ragvue.src.metrics.clarity",
    "model_env": "CLARITY_MODEL",
    "temp_env": "CLARITY_TEMPERATURE",
    "judges": [
        {"name": "cl_gpt4omini_t00_run1", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "cl_gpt4omini_t00_run2", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "cl_gpt4omini_t20",      "model": "gpt-4o-mini",   "temperature": 0.2},
        {"name": "cl_gpt4omini_t50",      "model": "gpt-4o-mini",   "temperature": 0.5},
        {"name": "cl_gpt41mini_t00",      "model": "gpt-4.1-mini",  "temperature": 0.0},
        {"name": "cl_gpt41mini_t30",      "model": "gpt-4.1-mini",  "temperature": 0.3},

    ],
}

TARGET_RETRIEVAL_COVERAGE = {
    "metric_name": "retrieval_coverage",
    "module_path": "ragvue.src.metrics.retrieval_coverage",
    "model_env": "RETRIEVAL_COVERAGE_MODEL",
    "temp_env": "RETRIEVAL_COVERAGE_TEMPERATURE",
    "judges": [
        {"name": "rc_gpt4omini_t00_run1", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "rc_gpt4omini_t00_run2", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "rc_gpt4omini_t20",      "model": "gpt-4o-mini",   "temperature": 0.2},
        {"name": "rc_gpt4omini_t50",      "model": "gpt-4o-mini",   "temperature": 0.5},
        {"name": "rc_gpt41mini_t00",      "model": "gpt-4.1-mini",  "temperature": 0.0},
        {"name": "rc_gpt41mini_t30",      "model": "gpt-4.1-mini",  "temperature": 0.3},

    ],
}

TARGET_RETRIEVAL_RELEVANCE = {
    "metric_name": "retrieval_relevance",
    "module_path": "ragvue.src.metrics.retrieval_relevance",
    "model_env": "RETRIEVAL_RELEVANCE_MODEL",
    "temp_env": "RETRIEVAL_RELEVANCE_TEMPERATURE",
    "judges": [
        {"name": "rr_gpt4omini_t00_run1", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "rr_gpt4omini_t00_run2", "model": "gpt-4o-mini",   "temperature": 0.0},
        {"name": "rr_gpt4omini_t20",      "model": "gpt-4o-mini",   "temperature": 0.2},
        {"name": "rr_gpt4omini_t50",      "model": "gpt-4o-mini",   "temperature": 0.5},
        {"name": "rr_gpt41mini_t00",      "model": "gpt-4.1-mini",  "temperature": 0.0},
        {"name": "rr_gpt41mini_t30",      "model": "gpt-4.1-mini",  "temperature": 0.3},

    ],
}

TARGETS = {
    "strict_faithfulness":   TARGET_STRICT_FAITHFULNESS,
    "answer_completeness":   TARGET_ANSWER_COMPLETENESS,
    "answer_relevance":      TARGET_ANSWER_RELEVANCE,
    "clarity":               TARGET_CLARITY,
    "retrieval_coverage":    TARGET_RETRIEVAL_COVERAGE,
    "retrieval_relevance":   TARGET_RETRIEVAL_RELEVANCE,
}

# Default target (can be overridden by wrappers)
CALIB_TARGET = TARGET_STRICT_FAITHFULNESS


def _import_metric(module_path: str):
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "evaluate"):
        raise RuntimeError(f"{module_path} has no evaluate(item)")
    return mod


def _coerce_score(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def _run_under_model(
    metric_mod,
    item: Dict[str, Any],
    model_env: str,
    model_value: str,
    temp_env: str | None,
    temperature: float | None,
) -> Dict[str, Any]:
    prev_model = os.getenv(model_env)
    prev_temp = os.getenv(temp_env) if temp_env else None
    try:
        os.environ[model_env] = model_value
        if temp_env is not None and temperature is not None:
            os.environ[temp_env] = str(temperature)

        out = metric_mod.evaluate(item)

    finally:
        # restore env
        if prev_model is None:
            os.environ.pop(model_env, None)
        else:
            os.environ[model_env] = prev_model

        if temp_env is not None:
            if prev_temp is None:
                os.environ.pop(temp_env, None)
            else:
                os.environ[temp_env] = prev_temp

    # Normalize: always return dict
    if not isinstance(out, dict):
        try:
            return {"score": float(out)}
        except Exception:
            return {"score": 0.0, "error": "non-dict return"}
    return out


def evaluate_with_target(item: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic calibration for a *specific* metric defined in cfg.
    Collects per-judge score + explanation/raw output.
    """
    metric_name = cfg["metric_name"]
    metric_mod = _import_metric(cfg["module_path"])
    model_env = cfg["model_env"]
    temp_env = cfg.get("temp_env")
    judges = cfg["judges"]

    sub_scores: Dict[str, float] = {}
    judge_outputs: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for jd in judges:
        j_name = jd["name"]
        j_model = jd["model"]
        j_temp = jd.get("temperature", None)
        try:
            out = _run_under_model(metric_mod, item, model_env, j_model, temp_env, j_temp)
            score = _coerce_score(out.get("score", 0.0))
            sub_scores[j_name] = score

            # keep explanation + raw info for this judge
            judge_outputs[j_name] = {
                "score": score,
                "explanation": out.get("explanation", ""),
                # everything else in case you want claims, etc.
                "raw": {k: v for k, v in out.items() if k not in ("score", "explanation")},
            }
        except Exception as e:
            errors[j_name] = str(e)

    if not sub_scores:
        return {
            "name": f"calibration_{metric_name}",
            "score": 0.0,
            "explanation": f"Calibration failed for {metric_name}; no valid scores",
            "details": {
                "metric_name": metric_name,
                "sub_scores": {},
                "judge_outputs": {},
                "min": 0.0,
                "max": 0.0,
                "spread": 1.0,
                "confidence": 0.0,
                "errors": errors,
            },
        }

    lo = min(sub_scores.values())
    hi = max(sub_scores.values())
    spread = hi - lo
    agreement = max(0.0, 1.0 - spread)

    explanation = (
        f"Calibration for {metric_name}: "
        f"{len(sub_scores)} judges, range [{lo:.2f}, {hi:.2f}], "
        f"spread={spread:.2f}, agreement={agreement:.2f}."
    )

    return {
        "name": f"calibration_{metric_name}",
        "score": agreement,
        "explanation": explanation,
        "details": {
            "metric_name": metric_name,
            "sub_scores": sub_scores,
            "judge_outputs": judge_outputs,  # <-- per-judge reason lives here
            "min": lo,
            "max": hi,
            "spread": spread,
            "confidence": agreement,
            "errors": errors,
        },
    }

# keep this for backwards compatibility
def evaluate(item: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_with_target(item, CALIB_TARGET)


IS_METRIC = True
