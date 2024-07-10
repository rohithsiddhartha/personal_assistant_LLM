import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import MagicMock, patch
from src.handlers import handle_follow_ups, classify_intent, match_exit_phrases
from src.utils import is_follow_up

def test_classify_intent():
    """
    Test the classify_intent function to ensure it correctly classifies the intent of a query.
    """
    with patch('src.utils.classifier') as mock_classifier:
        mock_classifier.return_value = {'labels': ['question']}
        query = "What is the weather today?"
        intent = classify_intent(query, mock_classifier)
        assert intent == 'question'

def test_match_exit_phrases():
    """
    Test the match_exit_phrases function to ensure it correctly identifies exit phrases.
    """
    follow_up_query = "Thanks, bye!"
    assert match_exit_phrases(follow_up_query) == True

def test_is_follow_up():
    """
    Test the is_follow_up function to ensure it correctly identifies if a text requires a follow-up action.
    """
    with patch('src.utils.classifier') as mock_classifier:
        mock_classifier.return_value = {'labels': ['Question']}
        text = "Do you need more details?"
        follow_up = is_follow_up(text, mock_classifier)
        assert follow_up == True

def test_handle_follow_ups():
    """
    Test the handle_follow_ups function to ensure it correctly handles follow-up interactions.
    """
    llm_manager = MagicMock()
    with patch('src.utils.classifier') as mock_classifier:
        mock_classifier.return_value = {'labels': ['question']}
        with patch('src.utils.is_follow_up', return_value=False) as mock_is_follow_up:
            context = "Test context"
            initial_query = "What is your name?"
            llm_manager.ask_question.return_value = "My name is Assistant."
            result = handle_follow_ups(llm_manager, "question", context, initial_query, mock_classifier)
            assert "Assistant: My name is Assistant." in result

