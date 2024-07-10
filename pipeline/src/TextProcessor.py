import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter

class TextProcessor:
    """
    A class to process text files, encode the texts using SentenceTransformers,
    and save the results to CSV files.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the TextProcessor with a specified model for encoding texts.
        """
        self.model = SentenceTransformer(model_name)
    
    def extract_text(self, file_path):
        """
        Extract text content from a given file path.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    
    def encode_texts(self, texts):
        """
        Encode a list of texts using the SentenceTransformer model.
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings
    
    def save_to_csv(self, file_path, texts, embeddings):
        """
        Save the texts and their embeddings to a CSV file.
        """
        data = {
            "Index": list(range(len(texts))),
            "Text": texts,
            "Embeddings": [emb.tolist() for emb in embeddings]
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def process_and_save_text(self, master_text, output_csv):
        """
        Process and save the text chunks and their embeddings to a CSV file.
        """
        chunks = self.text_splitter.split_text(master_text)
        embeddings = self.encode_texts(chunks)
        self.save_to_csv(output_csv, chunks, embeddings)
        # print(f"Chunks and embeddings saved to {output_csv}")
    
    def process_directory(self, directory):
        """
        Process all text files in a directory, combine the texts, and save the results to a CSV file.
        """
        master_text = ''
        output_csv = os.path.join(directory, "total_profile_data.csv")
        self.text_splitter = CharacterTextSplitter(chunk_size=300, 
                                                   chunk_overlap=100,
                                                   separator=" ")
                                                #    separator="\n")

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt") and not file.endswith("summary.txt"):
                    file_path = os.path.join(root, file)
                    text = self.extract_text(file_path)
                    print(f"Processing {file_path}...")

                    master_text +=f'\n {text} \n'

        self.process_and_save_text(master_text, output_csv)
        print(f"Full user Database is created at {output_csv}")
        return master_text

    def process_profile(self, directory):
        """
        Process summary text files in a directory and save the results to a CSV file.
        """
        master_text = ''
        output_csv = os.path.join(directory, "summary_profile.csv")
        # print(output_csv)
        self.text_splitter = CharacterTextSplitter(chunk_size=300, 
                                                   chunk_overlap=100,
                                                #    separator=" ")
                                                   separator="\n")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith("summary.txt"):
                    print("File", file)
                    file_path = os.path.join(root, file)
                    text = self.extract_text(file_path)
                    print(f"Processing {file_path}...")

                    master_text +=f'\n {text} \n'

        self.process_and_save_text(master_text, output_csv)
        print(f"Summary profile database  is created at {output_csv}")
        os.environ['SUMMARY_DB'] = output_csv
        return master_text
