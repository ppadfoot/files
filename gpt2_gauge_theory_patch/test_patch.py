from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PATCH = ROOT / "patch"
HELPER = PATCH / "notebooks" / "gauge_covariant_theory" / "gauge_theory_utils.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("gauge_theory_utils", HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ast.parse(HELPER.read_text(encoding="utf-8"))
    notebooks = sorted((PATCH / "notebooks" / "gauge_covariant_theory").glob("[0-1][0-9]_*.ipynb"))
    assert len(notebooks) == 11, len(notebooks)
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        for cell in payload["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    os.environ["GAUGE_SMOKE_TEST"] = "1"
    helper = load_helper()
    c = helper.smoke_covector()
    assert float(c.relative_l2_error.max()) < 1e-12
    m = helper.smoke_mixture().iloc[0]
    assert float(m.predicted_measured_distance) < 1e-12
    assert float(m.original_transformed_distance) > 0.01
    d = helper.smoke_optimizer_defect().set_index("optimizer").defect
    assert float(d["sophia_unsaturated"]) < 1e-20
    assert float(d["sgd"]) > 0.01
    s = helper.smoke_sophia_saturation()
    assert np.isfinite(s.defect).all()
    spec = helper.smoke_spectrum()
    assert float(spec.abs_diff.max()) < 1e-8
    flow = helper.smoke_batch_flow()
    assert abs(float(flow.fitted_gamma.iloc[0]) - 0.5) < 0.08
    print("static and mathematical smoke tests: PASS")


if __name__ == "__main__":
    main()
