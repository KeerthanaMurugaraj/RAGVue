
import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from typing import Callable, Any, Dict, List, Tuple

_DISCOVERY_ERRORS: Dict[str, str] = {}

# Modules we never want to register as metrics
_DENYLIST = {
    "__init__",
    "metrics_loader",
    "base",
    "base_judge",
    "utils",
    "env",
    "aspects",
    "smoke",
}


def _import_module_by_path(mod_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    raise ImportError(f"Cannot import {mod_name} from {path}")

def _lazy_class_metric(cls, fallback_name: str) -> Tuple[str, Callable[[dict], dict]]:
    """Wrap a class with an evaluate(self,item) into a lazy factory so we don't
    instantiate it during discovery (avoids requiring API keys/etc)."""
    instance = None
    name = getattr(cls, "name", fallback_name)
    def _call(item: dict) -> dict:
        nonlocal instance
        if instance is None:
            instance = cls()
        return instance.evaluate(item)
    return name, _call

def _collect_metric_callable(mod: Any, fallback_name: str, module_key: str) -> Dict[str, Callable[[dict], dict]]:
    """
    Collect a callable metric from a module.

    Rules:
      - If module sets IS_METRIC = False, skip it.
      - Prefer a module-level `evaluate(item)` function (simple + explicit).
      - Optionally allow classes only if they explicitly opt-in with IS_METRIC = True.
        (This avoids picking up BaseJudge or abstract helpers by accident.)
    """
    metrics: Dict[str, Callable[[dict], dict]] = {}

    # Module explicitly opts-out
    if getattr(mod, "IS_METRIC", True) is False:
        return metrics

    # 1) Prefer a module-level evaluate(item)
    fn = getattr(mod, "evaluate", None)
    if callable(fn):
        metrics[fallback_name] = fn
        return metrics

    # 2) Opt-in class-style (only classes with IS_METRIC=True)
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and getattr(obj, "IS_METRIC", False) is True and hasattr(obj, "evaluate"):
            try:
                name, wrapped = _lazy_class_metric(obj, fallback_name)
                metrics[name] = wrapped
            except Exception as e:
                _DISCOVERY_ERRORS[module_key] = f"class wrap failed: {e}"
                continue

    return metrics


def load_metrics(package: str = "ragvue.src.metrics") -> Dict[str, Callable[[dict], dict]]:
    """Auto-discover metrics modules exposing either `evaluate(item)` or a class with `evaluate`.
    Returns a dict mapping metrics name -> callable(item)->dict
    """
    metrics: Dict[str, Callable[[dict], dict]] = {}

    # Primary: import package and iterate with pkgutil
    try:
        pkg = importlib.import_module(package)
        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.ispkg or mod.name.startswith("_") or mod.name in _DENYLIST:
                continue
            mod_name = f"{package}.{mod.name}"

            try:
                m = importlib.import_module(mod_name)
            except Exception as e:
                _DISCOVERY_ERRORS[mod_name] = f"import failed: {e}"
                continue
            metrics.update(_collect_metric_callable(m, mod.name, mod_name))
    except Exception as e:
        _DISCOVERY_ERRORS[package] = f"package import failed: {e}"

    # Fallback: scan files on disk
    try:
        if 'pkg' not in locals():
            pkg = importlib.import_module(package)
        for p in pkg.__path__:
            base = Path(p)
            for py in base.glob("*.py"):
                stem = py.stem
                if stem == "__init__" or stem in _DENYLIST or stem.startswith("_"):
                    continue
                mod_name = f"{package}.{stem}"
                if mod_name in metrics or stem in metrics:
                    continue

                try:
                    m = _import_module_by_path(mod_name, py)
                except Exception as e:
                    _DISCOVERY_ERRORS[mod_name] = f"path import failed: {e}"
                    continue
                metrics.update(_collect_metric_callable(m, py.stem, mod_name))
    except Exception as e:
        _DISCOVERY_ERRORS[f"{package}.__path__scan"] = f"scan failed: {e}"

    return metrics

def select_metrics(requested: List[str], available: Dict[str, Any]) -> Dict[str, Any]:
    if not requested:
        return available
    selected = {}
    for r in requested:
        if r in available:
            selected[r] = available[r]
    return selected

def discovery_errors() -> Dict[str, str]:
    """Return import/wrap errors collected during discovery."""
    # trigger a discovery to populate errors
    load_metrics()
    return dict(_DISCOVERY_ERRORS)
