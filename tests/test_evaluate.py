import json

import cv2
import numpy as np

from scripts import evaluate

# Inner (white) ellipse axes; outer leaf is 100 x 150, so the true variegation is the area ratio.
INNER = {"a.jpg": (40, 60), "b.jpg": (50, 75), "c.jpg": (60, 90), "d.jpg": (70, 105)}


def write_dataset(tmp_path, white_bgr):
    rows = ["file,leaves,variegation_pct,notes"]
    for name, (ax, ay) in INNER.items():
        bgr = np.full((400, 400, 3), (60, 90, 140), np.uint8)
        cv2.ellipse(bgr, (200, 200), (100, 150), 0, 0, 360, (50, 130, 40), -1)
        cv2.ellipse(bgr, (200, 200), (ax, ay), 0, 0, 360, white_bgr, -1)
        cv2.imwrite(str(tmp_path / name), bgr)
        rows.append(f"{name},1,{100 * ax * ay / (100 * 150):.1f},")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n")


def test_run_reports_errors_and_caches_masks(tmp_path):
    write_dataset(tmp_path, white_bgr=(230, 240, 240))
    evaluate.main(["run", str(tmp_path)])

    summary = json.loads((tmp_path / "eval" / "summary.json").read_text())
    assert summary["leaf_count"] == {"n": 4, "exact": 1.0, "mae": 0}
    assert summary["variegation"]["mae"] < 3
    assert summary["grade_agreement"] == 1.0
    assert len(list((tmp_path / ".cache").glob("*.npz"))) == 4
    assert (tmp_path / "eval" / "results.csv").read_text().count("\n") == 5


def test_calibrate_recovers_dull_white_variegation(tmp_path):
    # Dull, grey-ish white (V=140) sits below the default white_min_v of 150.
    write_dataset(tmp_path, white_bgr=(140, 140, 140))
    evaluate.main(["run", str(tmp_path)])
    result = evaluate.cmd_calibrate(tmp_path)

    assert result["baseline_mae"] > 20
    assert result["tuned_mae_cross_validated"] < 3
    assert result["thresholds"]["white_min_v"] == 130
    assert json.loads((tmp_path / "eval" / "thresholds.json").read_text()) == result["thresholds"]
