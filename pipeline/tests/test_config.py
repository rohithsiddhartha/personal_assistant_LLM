import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from src.config import DEVICE, CLASSIFIER_MODEL, SUMMARY_DB

def test_device():
    assert DEVICE in ["mps", "cpu"]

def test_classifier_model():
    assert CLASSIFIER_MODEL == "facebook/bart-large-mnli"

def test_summary_db():
    assert isinstance(SUMMARY_DB, str) or SUMMARY_DB is None
