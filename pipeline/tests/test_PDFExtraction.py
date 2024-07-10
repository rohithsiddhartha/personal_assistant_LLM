import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import patch, MagicMock
from src.PDFExtraction import PDFExtraction

def test_extract_text():
    pdf_extractor = PDFExtraction('dummy.pdf', 'output_dir')
    mock_fitz_open = MagicMock()
    mock_fitz_doc = MagicMock()
    mock_fitz_open.return_value = mock_fitz_doc
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Sample text"
    mock_fitz_doc.__iter__.return_value = iter([mock_page])
    
    with patch('fitz.open', mock_fitz_open):
        text = pdf_extractor.extract_text()
        assert isinstance(text, str)
        assert text == "Sample text"

def test_extract_images():
    pdf_extractor = PDFExtraction('dummy.pdf', 'output_dir')
    with patch('fitz.open', return_value=MagicMock()) as mock_fitz:
        pdf_extractor.extract_images()
        assert mock_fitz.called

def test_extract_tables():
    pdf_extractor = PDFExtraction('dummy.pdf', 'output_dir')
    with patch('pdfplumber.open', return_value=MagicMock()) as mock_pdfplumber:
        pdf_extractor.extract_tables()
        assert mock_pdfplumber.called
