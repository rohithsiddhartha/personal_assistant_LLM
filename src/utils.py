from transformers import pipeline
import re
from config import CLASSIFIER_MODEL, DEVICE
import validators
import os

# Initialize the classifier pipeline with the specified model and device
classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=0 if DEVICE == "mps" else -1)

def get_valid_input(prompt, valid_options, max_retries=3):
    """
    Prompt the user for input and validate against a list of valid options.
    """
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
    """
    Classify the intent of a given query using a zero-shot classification model.
    """
    labels = ['save', 'question', 'update', 'delete', 'suggestion', 'emotion']
    result = classifier(query, labels)
    return result['labels'][0]

def is_follow_up(text, classifier):
    """
    Determine if a given text requires a follow-up action using a zero-shot classification model.
    """
    labels = ["Question", "Response", "Further Action Required"]
    result = classifier(text, labels)
    return result['labels'][0] in ["Question", "Further Action Required"]

def match_exit_phrases(follow_up_query):
    """
    Check if the follow-up query contains any exit phrases.
    """
    exit_phrases = re.compile(r"\b(thank you|thanks|no thanks|i'm good|no, i'm good|bye|goodbye|exit|no)\b", re.IGNORECASE)
    return exit_phrases.search(follow_up_query) is not None


def get_user_inputs(prompt, input_type='url'):
    """
    Prompt the user for multiple inputs until an empty input is received.
    """
    user_inputs = []
    while True:
        user_input = input(prompt).strip()
        if user_input:
            if input_type == 'url' and validators.url(user_input):
                user_inputs.append(user_input)
            elif input_type == 'file' and os.path.isfile(user_input):
                user_inputs.append(user_input)
            else:
                print(f"Invalid {input_type}. Please enter a valid {input_type}.")
        else:
            break
    return user_inputs

