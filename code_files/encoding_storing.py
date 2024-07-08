import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import numpy as np

def extract_text(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def encode_texts(texts, model_name='all-MiniLM-L6-v2'):

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def save_to_csv(file_path, texts, embeddings):

    data = {
        "Index": list(range(len(texts))),
        "Text": texts,
        "Embeddings": [emb.tolist() for emb in embeddings]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def process_and_save_text(file_path, output_csv):
    text = extract_text(file_path)
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = encode_texts(chunks)
    
    save_to_csv(output_csv, chunks, embeddings)
    print(f"Chunks and embeddings saved to {output_csv}")

if __name__ == "__main__":
    input_file_path = "extracted_html_files/saivineethkumar.github.io/extracted_text.txt"
    
    output_csv_path = "extracted_html_files/saivineethkumar.github.io/chunks_and_embeddings.csv"
    
    process_and_save_text(input_file_path, output_csv_path)
