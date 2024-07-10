import os
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CLASSIFIER_MODEL = "facebook/bart-large-mnli"
LLM_MODEL = "gpt-3.5-turbo-0125"  # Example model name, update as needed
