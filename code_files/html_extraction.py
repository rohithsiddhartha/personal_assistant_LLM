import asyncio
from requests_html import AsyncHTMLSession
import nest_asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path

# Apply nest_asyncio if running in a Jupyter notebook or interactive environment
if asyncio.get_event_loop().is_running():
    nest_asyncio.apply()

# Create the asynchronous HTML session
session = AsyncHTMLSession()

# Function to extract a safe folder name from a URL
def url_to_folder_name(url):
    parsed_url = urlparse(url)
    folder_name = parsed_url.netloc + parsed_url.path.replace("/", "_")
    return folder_name.strip('_')

# Define the main function to fetch and save HTML content
async def fetch_and_save_html(url):
    # Fetch the page
    r = await session.get(url)

    # Render the page
    await r.html.arender(sleep=1, keep_page=True, scrolldown=1)

    # Get the entire HTML content
    html_content = r.html.html

    # Create directory for saving the files
    folder_name = url_to_folder_name(url)
    base_dir = Path(f"extracted_html_files/{folder_name}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save the HTML content to a file
    rendered_file_path = base_dir / 'rendered_page.html'
    with open(rendered_file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    print(f"HTML content saved to '{rendered_file_path}'.")

    return rendered_file_path

async def clean_html(rendered_file_path):
    # Load the saved HTML content using BeautifulSoup
    with open(rendered_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Example of cleaning: remove all script and style tags
    for script in soup(["script", "style"]):
        script.decompose()

    # Save the cleaned content
    cleaned_file_path = rendered_file_path.parent / 'cleaned_page.html'
    with open(cleaned_file_path, 'w', encoding='utf-8') as file:
        file.write(str(soup.prettify()))
    print(f"Cleaned HTML content saved to '{cleaned_file_path}'.")

# Define a function to run the entire process
async def process_url(url):
    rendered_file_path = await fetch_and_save_html(url)
    await clean_html(rendered_file_path)

# Example usage
url = 'https://deshraj.xyz/'  # Replace with the URL you want to test
# url = """https://github.com/UMass-Rescue/IntelligentInformationExtractor/blob/main/model.py"""
# url = 'https://taranjeet.co/about/'asyncio.run(process_url(url))

asyncio.run(process_url(url))