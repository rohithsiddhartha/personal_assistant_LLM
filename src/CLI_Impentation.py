import asyncio
import os
from HTMLExtraction import HTMLExtraction
from PDFExtraction import PDFExtraction
from TextProcessor import TextProcessor
from transformers import pipeline
from Operations import DataManager
from LLMManager import LLMManager
from ProfileProcessor import ProfileProcessor

import torch
import re

device = "mps" if torch.backends.mps.is_available() else "cpu"
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device == "mps" else -1)
# follow_up_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device == "mps" else -1)

def get_valid_input(prompt, valid_options, max_retries=3):
    """To check if the user input is one of the valid options or not"""
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
    """to classifiy the internt of the user, Zero-shot classification is used"""
    labels = ['save', 'question', 'update', 'delete', 'suggestion', 'emotion']
    result = classifier(query, labels)
    return result['labels'][0]

def get_user_inputs(prompt):
    """To get the user inputs"""
    user_inputs = []
    while True:
        user_input = input(prompt).strip()
        if user_input:
            user_inputs.append(user_input)
        else:
            break
    return user_inputs

def is_follow_up(text):
    # labels = ["further assist", "question", "follow-up", "response"]
    labels = ["Question", "Response", "Further Action Required"]

    result = classifier(text, labels)
    print("Result of agent response", result['labels'])
    print(result['labels'][0] in ["Inquiry", "Question", "Further Action Required"])
    # return result['labels'][0] in ["follow-up", "question"]
    return result['labels'][0] in ["Inquiry", "Question", "Further Action Required"]

def match_exit_phrases(follow_up_query):
    exit_phrases = re.compile(r"\b(thank you|thanks|no thanks|i\'m good|no, i\'m good|bye|goodbye|exit|no)\b", re.IGNORECASE)
    return exit_phrases.search(follow_up_query) is not None

def handle_follow_ups(llm_manager,intent, context, initial_query):
    conversation_history = [f"Context: {context}, '\n', User: {initial_query}"]
    if intent=="question":
        response = llm_manager.ask_question(context, initial_query)
    else:
        response = llm_manager.ask_suggestion(context, initial_query)
    conversation_history.append(f"Assistant: {response}")
    print("\n\nAssistant Response: \n\n", response)

    while is_follow_up(response):
        follow_up_query = input("Assistant asked a follow-up question. Your response: ").strip()
        if match_exit_phrases(follow_up_query):
            print("Thank you. We can proceed for next query")
            break
        conversation_history.append(f"User: {follow_up_query}")
        context = "\n\n".join(conversation_history)
        print("*"*100)
        print("History", conversation_history)
        print("*"*100)
        print("user query", context)
        intent = classify_intent(follow_up_query)
        print("Intent of the follow up input query:", intent)
        if intent=="question":
            response = llm_manager.ask_question(context, initial_query)
        else:
            response = llm_manager.ask_suggestion(context, initial_query)
        conversation_history.append(f"Assistant: {response}")
        print("\n\nAssistant Response: \n\n", response)
    return conversation_history

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
    # output_csv = os.path.join(user_dir, "summary.csv")
    print("Processing all text files to create database")
    text_processor.process_directory(user_dir)

    print("Creating summary profile")
    # llm_manager = LLMManager(model="gpt-4o")
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
        
        intent = classify_intent(query)
        print(f"Intent classified as: {intent}")
        
        validation = get_valid_input(f"Is the intent '{intent}' correct? (yes/no): ", ['yes', 'no'])
        if validation == 'no':
            intent = get_valid_input("Please specify the intent (save, retrieve, update, delete): ", ['save', 'retrieve', 'update', 'delete' , 'suggestion', 'emotion'])

        if intent == 'save':
            updated_query = llm_manager.process_query(query)
            manager.save_entry(updated_query)
        elif intent == 'question':
            results = manager.retrieve_entries(query, k=5)
            print("Retrieved entries:")
            for result in results:
                print(result)
            context = "\n\n".join(results)
            conversation_history = handle_follow_ups(llm_manager, intent, context, query)
        elif intent == 'update':
            manager.update_entry(query)
        elif intent == 'delete':
            manager.delete_entry(query)
        elif intent in ['suggestion', 'emotion']:
            results = manager.retrieve_entries(query, k=5)
            context = "\n\n".join(results)
            conversation_history = handle_follow_ups(llm_manager, intent, context, query)
        elif intent in ['exit', 'quit']:
            print("Goodbye!")
            break
        else:
            print("Unknown intent. Please try again.")

if __name__ == "__main__":
    main()
