import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter

class TextProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def extract_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    
    def encode_texts(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings
    
    def save_to_csv(self, file_path, texts, embeddings):
        data = {
            "Index": list(range(len(texts))),
            "Text": texts,
            "Embeddings": [emb.tolist() for emb in embeddings]
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def process_and_save_text(self, master_text, output_csv):
        chunks = self.text_splitter.split_text(master_text)
        embeddings = self.encode_texts(chunks)
        self.save_to_csv(output_csv, chunks, embeddings)
        print(f"Chunks and embeddings saved to {output_csv}")
    
    def process_directory(self, directory, output_csv):
        all_texts = []
        all_embeddings = []
        master_text = ''
        self.text_splitter = CharacterTextSplitter(chunk_size=250, 
                                                   chunk_overlap=100,
                                                   separator=" ")
                                                #    separator=" \n")

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    text = self.extract_text(file_path)
                    print(f"Processing {file_path}...")

                    master_text +=f'\n {text} \n'

        self.process_and_save_text(master_text, output_csv)
        print(f"All chunks and embeddings saved to {output_csv}")
        return master_text
