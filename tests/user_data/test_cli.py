from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_cli_outer_cv_smoke(tmp_path):
    out = tmp_path / "parts"
    cmd = [
        sys.executable, str(ROOT / "tools" / "prepare_user_dataset.py"), "outer-cv",
        "--input", str(ROOT / "examples" / "user_dataset" / "toy_dataset.csv"),
        "--class-column", "class", "--numeric-method", "mdlp",
        "--outer-folds", "2", "--output-dir", str(out),
    ]
    subprocess.run(cmd, check=True)
    for name in ["Dataset1.dat", "1.dat", "Classes.dat", "class_dictionary.csv",
                 "preprocessing_fold_1.json", "preprocessing_manifest.json"]:
        assert (out / name).exists()


def test_cli_transactional_cv_smoke(tmp_path):
    source = tmp_path / "toy.dat"
    source.write_text("1 5 77\n2 6 77\n3 7 77\n4 8 77\n5 9 78\n6 10 78\n7 11 78\n8 12 78\n", encoding="utf-8")
    out = tmp_path / "parts_tx"
    cmd = [
        sys.executable, str(ROOT / "tools" / "prepare_user_dataset.py"), "transactional-cv",
        "--input", str(source), "--outer-folds", "2", "--seed", "123", "--output-dir", str(out),
    ]
    subprocess.run(cmd, check=True)
    assert (out / "Dataset1.dat").exists()
    assert (out / "1.dat").exists()
    assert (out / "Classes.dat").read_text(encoding="utf-8") == "77 4\n78 4\n"
