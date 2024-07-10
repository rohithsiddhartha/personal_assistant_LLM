import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import patch, MagicMock, mock_open
from src.ProfileProcessor import ProfileProcessor

def test_collect_texts():
    processor = ProfileProcessor(MagicMock(), 'test_dir')
    with patch('os.walk', return_value=[('test_dir', [], ['test.txt'])]):
        with patch('builtins.open', mock_open(read_data="Test content")):
            text = processor.collect_texts()
            assert text.strip() == "Test content"

def test_process_profile():
    llm_manager = MagicMock()
    processor = ProfileProcessor(llm_manager, 'test_dir')
    with patch.object(processor, 'collect_texts', return_value="Test content"):
        llm_manager.process_profile.return_value = '{"Section": {"Attribute": "Value"}}'
        with patch('builtins.open', mock_open()):
            processor.process_profile()
            assert llm_manager.process_profile.called

def test_write_profile_to_file():
    processor = ProfileProcessor(MagicMock(), 'test_dir')
    profile_dict = {"Section": {"Attribute": "Value"}}
    with patch('builtins.open', mock_open()) as mocked_file:
        processor.write_profile_to_file(profile_dict)
        mocked_file.assert_called_once_with(os.path.join('test_dir', 'user_profile_summary.txt'), 'w')
        handle = mocked_file()
        handle.write.assert_called_once()
