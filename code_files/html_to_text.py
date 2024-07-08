from bs4 import BeautifulSoup

def load_html(filename):
    """Load HTML content from a file."""
    with open(filename, 'r', encoding='utf-8') as file:
        html_content = file.read()
    return html_content

def parse_text_from_html(html_content):
    """Parse text information from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract text from paragraphs
    paragraphs = soup.find_all('p')
    paragraph_texts = [p.get_text() for p in paragraphs]

    # Extract text from headings (h1, h2, h3, etc.)
    headings = []
    for i in range(1, 7):
        for h in soup.find_all(f'h{i}'):
            headings.append(h.get_text())

    # Extract text from lists
    lists = []
    for ul in soup.find_all('ul'):
        lists.append([li.get_text() for li in ul.find_all('li')])
    
    # Combine extracted text
    extracted_text = {
        "paragraphs": paragraph_texts,
        "headings": headings,
        "lists": lists
    }

    return extracted_text

def save_extracted_text(extracted_text, filename):
    """Save the extracted text to a file."""
    with open(filename, 'w', encoding='utf-8') as file:
        for key, texts in extracted_text.items():
            file.write(f"### {key.upper()} ###\n\n")
            if isinstance(texts, list):
                for text in texts:
                    if isinstance(text, list):
                        for item in text:
                            file.write(f"- {item}\n")
                    else:
                        file.write(f"{text}\n")
            file.write("\n\n")
    print(f"Extracted text saved to '{filename}'.")

# Example usage
html_filename = 'extracted_html_files/deshraj.xyz/cleaned_page.html'  # Replace with your HTML file name
extracted_text_filename = 'extracted_html_files/deshraj.xyz/extracted_text.txt'  # Output file for the extracted text

# Load HTML content
html_content = load_html(html_filename)

# Parse text information from the HTML content
extracted_text = parse_text_from_html(html_content)

# Save the extracted text to a file
save_extracted_text(extracted_text, extracted_text_filename)
