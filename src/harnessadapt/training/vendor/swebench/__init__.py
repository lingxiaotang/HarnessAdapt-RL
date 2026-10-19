__version__ = "5.0.2"

_LAZY_IMPORTS = {
    "SWEbenchInstance": ("swebench.types", "SWEbenchInstance"),
    "compute_fail_to_pass": ("swebench.harness.grading", "compute_fail_to_pass"),
    "compute_pass_to_pass": ("swebench.harness.grading", "compute_pass_to_pass"),
    "get_eval_report": ("swebench.harness.grading", "get_eval_report"),
    "get_resolution_status": ("swebench.harness.grading", "get_resolution_status"),
    "ResolvedStatus": ("swebench.harness.grading", "ResolvedStatus"),
    "TestStatus": ("swebench.harness.grading", "TestStatus"),
    "PARSER_REGISTRY": ("swebench.harness.log_parsers", "PARSER_REGISTRY"),
}

__all__ = [
    "PARSER_REGISTRY",
    "ResolvedStatus",
    "SWEbenchInstance",
    "TestStatus",
    "__version__",
    "compute_fail_to_pass",
    "compute_pass_to_pass",
    "get_eval_report",
    "get_resolution_status",
]


def __getattr__(name: str):
    try:
        module_path, attr = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
