from transformers import pipeline
import torch

# Load a zero-shot classification model
device = "mps" if torch.backends.mps.is_available() else "cpu"

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)



def classify_intent(query):
    labels = ['save', 'retrieve', 'update', 'delete']
    result = classifier(query, labels)
    return result['labels'][0]


# Example usage
user_query = "Save my birthday as October 10th."
response = classify_intent(user_query)
print(response)  # Output based on the operation
