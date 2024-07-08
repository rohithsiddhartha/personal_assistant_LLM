from bs4 import BeautifulSoup

def extract_text_from_html(html_content):
    """Extract text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    return text
