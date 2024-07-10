from handlers import get_user_inputs, handle_follow_ups, classify_intent
from HTMLExtraction import HTMLExtraction
from PDFExtraction import PDFExtraction
from TextProcessor import TextProcessor
from transformers import pipeline
from Operations import DataManager
from LLMManager import LLMManager
from ProfileProcessor import ProfileProcessor

import torch
import os 
import asyncio
import shutil


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

def main():
    print("Welcome to the Building Personal Assistant!")


    while True:
        username = input("Enter your username: ").strip()
        user_dir = os.path.join(username, "extracted_contents")
        
        if os.path.exists(user_dir):
            add_info = get_valid_input("Username exists. Would you like to add information to the DB? (yes/no): ", ['yes', 'no'])
            if add_info == 'yes':
                break
            else:
                continue_with_existing = get_valid_input("Would you like to continue with the existing DB? (yes/no): ", ['yes', 'no'])
                if continue_with_existing == 'yes':
                    break
                else:
                    delete_db = get_valid_input("Would you like to delete the existing DB? (yes/no): ", ['yes', 'no'])
                    if delete_db == 'yes':
                        shutil.rmtree(user_dir)
                        create_new = get_valid_input("Would you like to create a new DB with the same username? (yes/no): ", ['yes', 'no'])
                        if create_new == 'yes':
                            break
                        else:
                            print("Please enter a different username.")
                    else:
                        print("Please enter a different username.")
        else:
            break
    
    # username = input("Enter your username: ").strip()
    # user_dir = os.path.join(username, "extracted_contents")
    os.makedirs(user_dir, exist_ok=True)
    
    pdf_paths = []
    urls = []

    if get_valid_input("Do you have PDF files to extract? (yes/no): ", ['yes', 'no']) == 'yes':
        pdf_paths = get_user_inputs("Enter the path to the PDF file (or press Enter to finish): ")
    
    if get_valid_input("Do you have URLs to extract? (yes/no): ", ['yes', 'no']) == 'yes':
        urls = get_user_inputs("Enter the URL (or press Enter to finish): ")
    
    
    if pdf_paths:
        for pdf_path in pdf_paths:
            pdf_extractor = PDFExtraction(pdf_path, base_extraction_dir=user_dir)
            extracted_text = pdf_extractor.extract_all()
            print(f"Extracted text from {pdf_path}:\n{extracted_text}\n")
            print(f"Extracted files are saved in: {pdf_extractor.base_dir}")
    
    if urls:
        save_intermediate = get_valid_input("Do you want to save intermediate files? (yes/no): ", ['yes', 'no']) == 'yes'
        loop = asyncio.get_event_loop()
        tasks = []
        for url in urls:
            html_extractor = HTMLExtraction(url, save_intermediate=save_intermediate, base_extraction_dir=user_dir)
            tasks.append(loop.create_task(html_extractor.extract_all()))
        loop.run_until_complete(asyncio.gather(*tasks))
    
    # Process all text files and save to a single CSV
    text_processor = TextProcessor()
    print("Processing all text files to create database")
    text_processor.process_directory(user_dir)

    print("Creating summary profile")
    llm_manager = LLMManager()
    profile_processor = ProfileProcessor(llm_manager, user_dir)
    profile_processor.process_profile()

    print("Processing summary profile file to create database")
    text_processor.process_profile(user_dir)

    print("Your personal assistant is now ready!")

    metric = 'mmr'
    try:
        manager = DataManager(csv_file=os.getenv("SUMMARY_DB"), metric=metric)
    except:
        print("Please run the script again to create the summary database.")
        return
    
    while True:
        query = input("Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        intent = classify_intent(query, classifier)
        print(f"Intent classified as: {intent}")
        
        validation = get_valid_input(f"Is the intent '{intent}' correct? (yes/no): ", ['yes', 'no'])
        if validation == 'no':
            intent = get_valid_input("Please specify the intent (save, retrieve, update, delete, suggestion, emotion, question): ", ['save', 'retrieve', 'update', 'delete' , 'suggestion', 'emotion', 'question'])

        if intent == 'save':
            updated_query = llm_manager.process_query(query)
            manager.save_entry(updated_query)
        elif intent == 'question':
            results = manager.retrieve_entries(query, k=5)
            print("Retrieved entries:")
            for result in results:
                print(result)
            context = "\n\n".join(results)
            conversation_history = handle_follow_ups(llm_manager, intent, context, query, classifier)
        elif intent == 'update':
            manager.update_entry(query)
        elif intent == 'delete':
            manager.delete_entry(query)
        elif intent in ['suggestion', 'emotion']:
            results = manager.retrieve_entries(query, k=5)
            context = "\n\n".join(results)
            conversation_history = handle_follow_ups(llm_manager, intent, context, query, classifier)
        elif intent in ['exit', 'quit']:
            print("Goodbye!")
            break
        else:
            print("Unknown intent. Please try again.")

if __name__ == "__main__":
    main()
