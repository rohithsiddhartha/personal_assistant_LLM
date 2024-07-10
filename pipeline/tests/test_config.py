import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from src.config import DEVICE, CLASSIFIER_MODEL, SUMMARY_DB

def test_device():
    """
    Test to ensure that the DEVICE variable is set to either 'mps' or 'cpu'.
    """
    assert DEVICE in ["mps", "cpu"]

def test_classifier_model():
    """
    Test to ensure that the CLASSIFIER_MODEL variable is set to 'facebook/bart-large-mnli'.
    """
    assert CLASSIFIER_MODEL == "facebook/bart-large-mnli"

def test_summary_db():
    """
    Test to ensure that the SUMMARY_DB variable is a string or None.
    """
    assert isinstance(SUMMARY_DB, str) or SUMMARY_DB is None
