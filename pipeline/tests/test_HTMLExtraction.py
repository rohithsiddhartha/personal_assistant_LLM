import pytest
import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import patch, AsyncMock
from src.HTMLExtraction import HTMLExtraction

@pytest.mark.asyncio
async def test_fetch_and_save_html():
    html_extractor = HTMLExtraction('https://example.com', save_intermediate=True)
    
    mock_response = AsyncMock()
    mock_response.html.arender = AsyncMock()
    mock_response.html.html = "<html><body><p>Hello</p></body></html>"
    
    with patch('requests_html.AsyncHTMLSession.get', new=AsyncMock(return_value=mock_response)) as mock_get:
        html_content = await html_extractor.fetch_and_save_html()
        assert html_content is not None

@pytest.mark.asyncio
async def test_clean_html():
    html_extractor = HTMLExtraction('https://example.com')
    html_content = "<html><body><script>test</script><p>Hello</p></body></html>"
    cleaned_html = await html_extractor.clean_html(html_content)
    assert '<script>' not in cleaned_html

@pytest.mark.asyncio
async def test_extract_text_from_html():
    html_extractor = HTMLExtraction('https://example.com')
    html_content = "<html><body><p>Hello</p></body></html>"
    text = await html_extractor.extract_text_from_html(html_content)
    assert text.strip() == "Hello"