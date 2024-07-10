import pytest
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.utils import classify_intent, is_follow_up, match_exit_phrases, get_valid_input, get_user_inputs

def test_classify_intent():
    classifier = MagicMock()
    classifier.return_value = {'labels': ['save']}
    query = "Please save this input"
    intent = classify_intent(query, classifier)
    assert intent == "save"

def test_is_follow_up():
    classifier = MagicMock()
    classifier.return_value = {'labels': ['Question']}
    text = "Do you need more details?"
    follow_up = is_follow_up(text, classifier)
    assert follow_up == True

def test_match_exit_phrases():
    follow_up_query = "Thanks, bye!"
    exit_match = match_exit_phrases(follow_up_query)
    assert exit_match == True

def test_get_valid_input(monkeypatch):
    inputs = iter(["invalid", "yes"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    valid_input = get_valid_input("Continue? (yes/no): ", ['yes', 'no'])
    assert valid_input == "yes"

def test_get_user_inputs(monkeypatch):
    inputs = iter(["input1", "input2", ""])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    user_inputs = get_user_inputs("Enter input: ")
    assert user_inputs == ["input1", "input2"]

if __name__ == "__main__":
    pytest.main()