import os
import torch
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CLASSIFIER_MODEL = "facebook/bart-large-mnli"
SUMMARY_DB = os.getenv("SUMMARY_DB")
