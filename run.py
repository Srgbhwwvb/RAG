#!/usr/bin/env python3

from __future__ import annotations

from glob import glob
from json import dumps, load
from os import environ
from pathlib import Path
from sys import argv, exit


MAX_MARK = 20.0
TEST_SPECS = (
    ("00_unittest_download_input", "download", 1.0),
    ("01_unittest_preprocessing_input", "preprocessing", 3.0),
    ("02_unittest_embedding_input", "embedding", 2.0),
    ("03_unittest_vector_db_ids_input", "vector_db_ids", 2.0),
    ("04_unittest_vector_db_query_input", "vector_db_query", 4.0),
    ("05_unittest_generation_input", "generation_artifact", 2.0),
    ("06_unittest_generation_q1_input", "generation_q1", 1.0),
    ("07_unittest_generation_q2_input", "generation_q2", 1.0),
    ("08_unittest_generation_q3_input", "generation_q3", 1.0),
    ("09_unittest_generation_q4_input", "generation_q4", 1.0),
    ("10_unittest_generation_q5_input", "generation_q5", 2.0),
)

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_test_file(data_dir: Path) -> Path:
    exact_test = data_dir / "test.py"
    if exact_test.exists():
        return exact_test

    test_files = sorted(data_dir.glob("test*.py"))
    if len(test_files) != 1:
        raise FileNotFoundError(f"Expected exactly one test file in {data_dir}")
    return test_files[0]


def run_single_test(data_dir: str, output_dir: str) -> None:
    from pytest import main

    del output_dir
    test_path = _find_test_file(Path(data_dir))
    exit(main(["-vv", "-s", "-p", "no:cacheprovider", str(test_path)]))


def check_test(data_dir: str) -> None:
    del data_dir


def grade(data_path: str) -> dict[str, float | str]:
    results_path = Path(data_path) / "results.json"
    results = load(results_path.open())

    total_grade = 0.0
    ok_count = 0
    for result, (_, _, weight) in zip(results, TEST_SPECS):
        if result.get("status") == "Ok":
            total_grade += weight
            ok_count += 1

    total_count = len(TEST_SPECS)
    description = f"{ok_count:02d}/{total_count:02d}"
    mark = total_grade / sum(weight for _, _, weight in TEST_SPECS) * MAX_MARK
    res = {"description": description, "mark": mark}
    if environ.get("CHECKER"):
        print(dumps(res))
    return res
if __name__ == "__main__":
    if environ.get("CHECKER"):
        if len(argv) != 4:
            print(f"Usage: {argv[0]} mode data_dir output_dir")
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == "run_single_test":
            run_single_test(data_dir, output_dir)
        elif mode == "check_test":
            check_test(data_dir)
        elif mode == "grade":
            grade(data_dir)
        else:
            print(f"Unknown mode: {mode}")
            exit(1)
    else:
        if len(argv) != 3:
            print(f"Usage: {argv[0]} test/unittest test_name")
            exit(0)

        tests_dir = SCRIPT_DIR / "tests"
        if not tests_dir.is_dir():
            print(
                "Directory `tests` not found\n"
                "Please create it and extract there all "
                "folders with tests from test_inputs.zip",
            )
            exit(1)

        mode = argv[1]
        test_name = argv[2]
        test_dirs = sorted(tests_dir.glob(f"[0-9][0-9]_{mode}_{test_name}_input"))
        if not test_dirs:
            print("Test not found")
            exit(0)

        from pytest import main

        exit(main(["-vv", "-x", "-s", "-p", "no:cacheprovider", str(_find_test_file(test_dirs[0]))]))
