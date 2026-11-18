import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "report" / "main" / "tools"))
from plate_history import diff_args, parse_notebook  # noqa: E402

LINE = "                   all       9463      13712      {}      {}      {}      {}\n"


def nb(tmp_path, cells):
    p = tmp_path / "x.ipynb"
    p.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return p


def test_parse_constants_and_epoch_lines(tmp_path):
    cells = [
        {"cell_type": "code", "source": ['BASE_MODEL = "Wiliv3.pt"\n', "EPOCHS = 3  # note\n", "NEW_LR0 = 0.0002\n",
                                         "BATCH = PER_GPU_BATCH * 2\n"], "outputs": []},
        {"cell_type": "code", "source": ["train()"], "outputs": [
            {"text": [LINE.format(0.966, 0.953, 0.979, 0.655)]},
            {"text": [LINE.format(0.98, 0.973, 0.988, 0.68), LINE.format(0.987, 0.978, 0.992, 0.703),
                      LINE.format(0.987, 0.978, 0.992, 0.703)]}]},
    ]
    r = parse_notebook(nb(tmp_path, cells))
    assert r["config"]["BASE_MODEL"] == "Wiliv3.pt" and r["config"]["EPOCHS"] == "3"
    assert r["config"]["NEW_LR0"] == "0.0002" and "BATCH" not in r["config"]
    assert r["epochs"] == [[0.966, 0.953, 0.979, 0.655], [0.98, 0.973, 0.988, 0.68], [0.987, 0.978, 0.992, 0.703]]
    assert r["final"] == [0.987, 0.978, 0.992, 0.703]
    assert (r["n_images"], r["n_instances"]) == (9463, 13712)


def test_diff_args():
    assert diff_args({"EPOCHS": "30", "IMG_SIZE": "768"}, {"EPOCHS": "50", "IMG_SIZE": "768", "LR": "0.01"}) == \
        [("EPOCHS", "30", "50"), ("LR", "not recorded", "0.01")]
