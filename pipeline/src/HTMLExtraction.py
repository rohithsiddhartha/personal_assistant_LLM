import asyncio
from requests_html import AsyncHTMLSession
import nest_asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path
import re

# Apply nest_asyncio if running in a Jupyter notebook or interactive environment
if asyncio.get_event_loop().is_running():
    nest_asyncio.apply()

class HTMLExtraction:
    """
    A class for extracting and processing HTML content from a URL.
    This class fetches, renders, cleans, and extracts text from HTML content,
    and optionally saves intermediate results.
    """
    def __init__(self, url, save_intermediate=False, base_extraction_dir='extracted_html_files'):
        self.url = url
        self.save_intermediate = save_intermediate
        self.session = AsyncHTMLSession()
        self.base_dir = self._create_base_dir(base_extraction_dir)
    
    def _create_base_dir(self, base_extraction_dir):
        """
        Create a base directory for storing intermediate and final extraction results.
        """
        folder_name = self._url_to_folder_name(self.url)
        base_dir = Path(base_extraction_dir) / folder_name
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def _url_to_folder_name(self, url):
        """
        Convert a URL to a valid folder name.
        """
        parsed_url = urlparse(url)
        folder_name = parsed_url.netloc + parsed_url.path.replace("/", "_")
        return folder_name.strip('_')

    async def fetch_and_save_html(self):
        """
        Fetch and render HTML content from the specified URL,
        and optionally save the rendered HTML to a file.
        """
        try:
            # Fetch the page
            r = await self.session.get(self.url)

            # Render the page
            await r.html.arender(sleep=1, keep_page=True, scrolldown=1)

            # Get the entire HTML content
            html_content = r.html.html

            if self.save_intermediate:
                # Save the HTML content to a file
                rendered_file_path = self.base_dir / 'rendered_page.html'
                with open(rendered_file_path, 'w', encoding='utf-8') as file:
                    file.write(html_content)
                print(f"HTML content saved to '{rendered_file_path}'.")
                return rendered_file_path
            else:
                return html_content
        except Exception as e:
            raise RuntimeError("Extraction request is blocked due to a network policy")

    async def clean_html(self, html_content):
        """
        Clean the HTML content by removing unnecessary tags,
        and optionally save the cleaned HTML to a file.
        """
        if isinstance(html_content, Path):
            # Load the saved HTML content using BeautifulSoup
            with open(html_content, 'r', encoding='utf-8') as file:
                html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Example of cleaning: remove all script and style tags
        for script in soup(["script", "style"]):
            script.decompose()

        cleaned_html_content = str(soup.prettify())
        
        if self.save_intermediate:
            # Save the cleaned content
            cleaned_file_path = html_content.parent / 'cleaned_page.html'
            with open(cleaned_file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_html_content)
            print(f"Cleaned HTML content saved to '{cleaned_file_path}'.")
            return cleaned_file_path
        else:
            return cleaned_html_content

    def _normalize_whitespace(self, text):
        """
        Remove redundant spaces and normalize newlines in the text.
        """
        # Replace multiple spaces or tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    async def extract_text_from_html(self, html_content):
        """
        Extract text from HTML content and normalize whitespace.
        """
        if isinstance(html_content, Path):
            with open(html_content, 'r', encoding='utf-8') as file:
                html_content = file.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator='\n')

        # Normalize whitespace
        normalized_text = self._normalize_whitespace(text)

        text_file = self.base_dir / 'extracted_text.txt'
        with open(text_file, 'w', encoding='utf-8') as file:
            file.write(normalized_text)
        
        print(f"Text extracted to file: {text_file}")
        return normalized_text

    async def extract_all(self):
        """
        Perform the complete extraction process: fetch, clean, and extract text from HTML content.
        """
        html_content = await self.fetch_and_save_html()
        cleaned_html_content = await self.clean_html(html_content)
        text = await self.extract_text_from_html(cleaned_html_content)
        return text

    async def close(self):
        """
        Close the asynchronous HTML session.
        """
        await self.session.close()

# Test the HTMLExtraction class
async def test_html_extraction(url, save_intermediate):
    html_extractor = HTMLExtraction(url, save_intermediate=save_intermediate)
    try:
        extracted_text = await html_extractor.extract_all()
        print(extracted_text)
    finally:
        await html_extractor.close()

if __name__ == "__main__":
    # Example usage for testing
    url = 'https://deshraj.xyz/'  # Replace with the URL you want to test
    save_intermediate = True
    asyncio.run(test_html_extraction(url, save_intermediate))
