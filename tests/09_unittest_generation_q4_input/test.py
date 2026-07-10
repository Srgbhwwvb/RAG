import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import DEFAULT_QUESTIONS, answer_matches, load_module


def test_answer_separate_peace_1918():
    generation_module = load_module("generation")
    saved_answers = generation_module.load_answers(generation_module.default_answers_path())

    question = DEFAULT_QUESTIONS[3]
    assert answer_matches(question, saved_answers[question])
