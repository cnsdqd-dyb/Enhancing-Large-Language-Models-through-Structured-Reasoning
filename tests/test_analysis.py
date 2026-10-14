import importlib.util
import json
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lighteval_aggregation(tmp_path):
    payload = {
        "config_general": {
            "generation_parameters": {"max_new_tokens": 4096, "temperature": 0.6}
        },
        "results": {
            "math_500": {"extractive_match": 0.75, "extractive_match_stderr": 0.02}
        },
    }
    for seed in range(2):
        (tmp_path / f"seed-{seed}.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = _load_script("analyze_evaluations").aggregate(tmp_path)
    assert rows == [{
        "max_new_tokens": 4096,
        "temperature": 0.6,
        "task": "math_500",
        "metric": "extractive_match",
        "mean": 0.75,
        "std": 0.0,
        "runs": 2,
    }]
