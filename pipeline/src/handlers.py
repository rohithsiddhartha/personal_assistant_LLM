import re
from transformers import pipeline
import torch
from utils import classify_intent, match_exit_phrases, is_follow_up

# def classify_intent(query, classifier):
#     labels = ['save', 'question', 'update', 'delete', 'suggestion', 'emotion']
#     result = classifier(query, labels)
#     return result['labels'][0]

# def get_user_inputs(prompt):
#     user_inputs = []
#     while True:
#         user_input = input(prompt).strip()
#         if user_input:
#             user_inputs.append(user_input)
#         else:
#             break
#     return user_inputs

# def is_follow_up(text, classifier):
#     labels = ["Question", "Response", "Further Action Required"]
#     result = classifier(text, labels)
#     return result['labels'][0] in ["Question", "Further Action Required"]

# def match_exit_phrases(follow_up_query):
#     exit_phrases = re.compile(r"\b(thank you|thanks|no thanks|i\'m good|no, i\'m good|bye|goodbye|exit|no)\b", re.IGNORECASE)
#     return exit_phrases.search(follow_up_query) is not None

def handle_follow_ups(llm_manager, intent, context, initial_query, classifier):
    conversation_history = [f"Context: {context}, '\n', User: {initial_query}"]
    if intent == "question":
        response = llm_manager.ask_question(context, initial_query)
    else:
        response = llm_manager.ask_suggestion(context, initial_query)
    conversation_history.append(f"Assistant: {response}")
    print("\n\nAssistant Response: \n\n", response)

    while is_follow_up(response, classifier):
        follow_up_query = input("Assistant asked a follow-up question. Your response: ").strip()
        if not follow_up_query:
            print("No input detected. Ending follow-up.")
            break
        if match_exit_phrases(follow_up_query):
            print("Thank you. We can proceed for next query")
            break
        conversation_history.append(f"User: {follow_up_query}")
        context = "\n\n".join(conversation_history)
        intent = classify_intent(follow_up_query)
        if intent == "question":
            response = llm_manager.ask_question(context, initial_query)
        else:
            response = llm_manager.ask_suggestion(context, initial_query)
        conversation_history.append(f"Assistant: {response}")
        print("\n\nAssistant Response: \n\n", response)
    return conversation_history
