import asyncio
import os
from HTMLExtraction import HTMLExtraction
from PDFExtraction import PDFExtraction
from TextProcessor import TextProcessor
from transformers import pipeline
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device == "mps" else -1)


def get_valid_input(prompt, valid_options, max_retries=3):
    attempts = 0
    while attempts < max_retries:
        user_input = input(prompt).strip().lower()
        if user_input in valid_options:
            return user_input
        print(f"Invalid input. Please enter one of {valid_options}.")
        attempts += 1
    print("Too many invalid attempts. Exiting.")
    return None

def classify_intent(query):
    labels = ['save', 'retrieve', 'update', 'delete']
    result = classifier(query, labels)
    return result['labels'][0]

def get_user_inputs(prompt):
    user_inputs = []
    while True:
        user_input = input(prompt).strip()
        if user_input:
            user_inputs.append(user_input)
        else:
            break
    return user_inputs

def main():
    print("Welcome to the Building Personal Assistant!")
    
    username = input("Enter your username: ").strip()
    user_dir = os.path.join(username, "extracted_contents")
    os.makedirs(user_dir, exist_ok=True)
    
    pdf_paths = []
    urls = []
    


    if get_valid_input("Do you have PDF files to extract? (yes/no): ", ['yes', 'no']) == 'yes':
        pdf_paths = get_user_inputs("Enter the path to the PDF file (or press Enter to finish): ")
    
    if get_valid_input("Do you have URLs to extract? (yes/no): ", ['yes', 'no']) == 'yes':
        urls = get_user_inputs("Enter the URL (or press Enter to finish): ")
    
    save_intermediate = get_valid_input("Do you want to save intermediate files? (yes/no): ", ['yes', 'no']) == 'yes'
    
    if pdf_paths:
        for pdf_path in pdf_paths:
            pdf_extractor = PDFExtraction(pdf_path, base_extraction_dir=user_dir)
            extracted_text = pdf_extractor.extract_all()
            print(f"Extracted text from {pdf_path}:\n{extracted_text}\n")
            print(f"Extracted files are saved in: {pdf_extractor.base_dir}")
    
    if urls:
        loop = asyncio.get_event_loop()
        tasks = []
        for url in urls:
            html_extractor = HTMLExtraction(url, save_intermediate=save_intermediate, base_extraction_dir=user_dir)
            tasks.append(loop.create_task(html_extractor.extract_all()))
        loop.run_until_complete(asyncio.gather(*tasks))
    
    # Process all text files and save to a single CSV
    text_processor = TextProcessor()
    output_csv = os.path.join(user_dir, "all_chunks_and_embeddings.csv")
    print("Processing all text files to create chunks and embeddings...")
    text_processor.process_directory(user_dir, output_csv)
    print(f"Database is created at {output_csv}")

    print("Your personal assistant is now ready!")

    # Prompt the user to ask a question
    while True:
        query = input("Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        intent = classify_intent(query)
        print(f"Intent classified as: {intent}")
if __name__ == "__main__":
    main()
