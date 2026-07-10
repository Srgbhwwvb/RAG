import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import (
    DEFAULT_QUESTIONS,
    load_module,
)


def test_answers_json_exists_and_contains_all_expected_questions():
    generation_module = load_module("generation")
    answers_path = generation_module.default_answers_path()

    assert answers_path.exists()
    saved_answers = generation_module.load_answers(answers_path)
    assert list(saved_answers) == list(DEFAULT_QUESTIONS)
