from transformers import pipeline
import re
from config import CLASSIFIER_MODEL, DEVICE

classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=0 if DEVICE == "mps" else -1)

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

def classify_intent(query, classifier):
    labels = ['save', 'question', 'update', 'delete', 'suggestion', 'emotion']
    result = classifier(query, labels)
    return result['labels'][0]

def is_follow_up(text, classifier):
    labels = ["Question", "Response", "Further Action Required"]
    result = classifier(text, labels)
    return result['labels'][0] in ["Question", "Further Action Required"]

def match_exit_phrases(follow_up_query):
    exit_phrases = re.compile(r"\b(thank you|thanks|no thanks|i'm good|no, i'm good|bye|goodbye|exit|no)\b", re.IGNORECASE)
    return exit_phrases.search(follow_up_query) is not None

def get_user_inputs(prompt):
    user_inputs = []
    while True:
        user_input = input(prompt).strip()
        if user_input:
            user_inputs.append(user_input)
        else:
            break
    return user_inputs
