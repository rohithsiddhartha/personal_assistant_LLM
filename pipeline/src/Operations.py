import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ast import literal_eval
import os

from LLMManager import LLMManager

class DataManager:
    def __init__(self, csv_file, model_name='sentence-transformers/all-MiniLM-L6-v2', metric="cosine"):
        self.csv_file = csv_file
        print(model_name)
        
        self.model_name = model_name
        print("self", self.model_name)

              
    
        self.model = SentenceTransformer(model_name)
        self.df = self.load_csv()
        self.metric = metric

    def load_csv(self):
        print("Loading CSV...")
        df = pd.read_csv(self.csv_file)
        # print(f"before literal eval - type= {type(df['Embeddings'][0])}", df['Embeddings'][0])
        df['Embeddings'] = df['Embeddings'].apply(literal_eval).apply(np.array)
        # print(f"after literal eval - type= {type(df['Embeddings'][0])}", df['Embeddings'][0])
        print("CSV Loaded.")
        return df

    def save_csv(self):
        print("Saving CSV...")
        self.df['Embeddings'] = self.df['Embeddings'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        self.df.to_csv(self.csv_file, index=False)
        print("CSV Saved.")

    def encode_text(self, text):
        print(f"Encoding text: {text[:30]}...")  # Print the first 30 characters of the text
        embedding = self.model.encode([text], convert_to_tensor=True).cpu().numpy()[0]
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        print(f"Text encoded. Embedding shape: {embedding.shape}")
        return embedding

    def save_entry(self, text):
        print(f"Saving new input: {text}")  
        new_embedding = self.encode_text(text)
        new_entry = pd.DataFrame([{'Text': text, 'Embeddings': new_embedding.tolist()}])
        self.df = pd.concat([self.df, new_entry], ignore_index=True)
        self.save_csv()
        print("Entry saved to profile.")

    def retrieve_entries(self, query, k=10, return_indices=False):
        print(f"Retrieving entries using {self.metric} metric for query: {query[:50]}...")  # Print the first 50 characters of the query
        query_embedding = self.encode_text(query)
        embeddings = np.vstack(self.df['Embeddings'].values)
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
            faiss.normalize_L2(query_embedding)
            index = faiss.IndexFlatIP(embeddings.shape[1]) 
        elif self.metric == 'mmr':
            return self.mmr(query_embedding, top_k=k)
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])
        
        index.add(embeddings)
        D, I = index.search(query_embedding.reshape(1, -1), k)
        if return_indices:
            return I[0], [self.df.iloc[i]['Text'] for i in I[0]]
        else:
            return [self.df.iloc[i]['Text'] for i in I[0]]

    def update_entry(self, query, k=1):
        retrieved_indices, top_k_texts = self.retrieve_entries(query, k, return_indices=True)
        while True:
            print("Top similar entries:")
            for i, text in enumerate(top_k_texts):
                print(f"{i + 1}: {text}")

            selection = input("Enter the number of the entry you want to update (or type 'more' to see more entries, 'cancel' to cancel): ").strip().lower()
            if selection == 'cancel':
                break
            elif selection == 'more':
                k += 3
                retrieved_indices, top_k_texts = self.retrieve_entries(query, k, return_indices=True)
                continue
            elif selection.isdigit() and 1 <= int(selection) <= len(top_k_texts):
                index_to_update = retrieved_indices[int(selection) - 1]
                updated_embedding = self.encode_text(query)
                self.df.at[index_to_update, 'Text'] = query
                self.df.at[index_to_update, 'Embeddings'] = updated_embedding.tolist()
                self.save_csv()
                print("Entry updated in CSV.")
                break
            else:
                print("Invalid selection. Please try again.")

        if selection == 'cancel':
            save_to_profile = input("Do you want to save this information to your profile? (yes/no): ").strip().lower()
            if save_to_profile == 'yes':
                self.save_entry(query)
            else:
                print("Update cancelled and entry not saved to profile.")

    def delete_entry(self, query, k=1):
        print(f"Deleting entry for query: {query[:50]}...")  # Print the first 50 characters of the query
        top_k_texts = self.retrieve_entries(query, k)
        print("Top 1 similar entry:")
        print(top_k_texts[0])
        
        confirmation = input("Do you want to delete this entry? (yes/no): ").strip().lower()
        if confirmation == 'yes':
            index_to_remove = self.df[self.df['Text'] == top_k_texts[0]].index[0]
            self.df = self.df.drop(index_to_remove).reset_index(drop=True)
            self.save_csv()
            print("Entry deleted from CSV.")
        else:
            print("Deletion cancelled.")

    def mmr(self, query_embedding, top_k=5, lambda_param=0.5):
        print("Running MMR...")
        doc_embeddings = np.vstack(self.df['Embeddings'].values)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        sim_scores = np.dot(doc_embeddings, query_embedding.T).flatten()
        
        selected = []
        selected_scores = []
        
        while len(selected) < top_k:
            if not selected:
                selected_idx = np.argmax(sim_scores)
            else:
                mmr_scores = []
                for idx in range(len(doc_embeddings)):
                    if idx not in selected:
                        diversity = max([np.dot(doc_embeddings[idx], doc_embeddings[sel_idx]) for sel_idx in selected])
                        mmr_score = lambda_param * sim_scores[idx] - (1 - lambda_param) * diversity
                        mmr_scores.append((mmr_score, idx))
                selected_idx = max(mmr_scores)[1]
            
            selected.append(selected_idx)
            selected_scores.append(sim_scores[selected_idx])
        
        print("MMR completed.")
        return [self.df.iloc[i]['Text'] for i in selected]

if __name__ == "__main__":
    csv_file = "/Users/rohithsiddharthareddy/Desktop/TakeHomeAssignments/personal_assistant_LLM/rohithsiddhartha/all_chunks_and_embeddings.csv"
    manager = DataManager(csv_file)
    llm_manager = LLMManager()  # Replace with your actual OpenAI API key

    
    while True:
        operation = input("Enter operation (save, retrieve, update, delete, exit): ").strip().lower()
        
        if operation == 'save':
            new_text = input("Enter the text to save: ")
            manager.save_entry(new_text)
        elif operation == 'retrieve':
            query = input("Enter the query text to retrieve similar entries: ")
            metric = input("Enter the metric (cosine, euclidean, mmr): ").strip().lower()
            results = manager.retrieve_entries(query, k=10, metric=metric)
            print("Retrieved entries:")
            for result in results:
                print(result)
            context = "\n\n".join(results)
            answer = llm_manager.ask_question(context, query)
            print("\n\nAssistant Response: \n\n", answer)
            
        elif operation == 'update':
            query = input("Enter the query text to find the entry to update: ")
            manager.update_entry(query)
        elif operation == 'delete':
            query = input("Enter the query text to find the entry to delete: ")
            manager.delete_entry(query)
        elif operation == 'exit':
            break
        else:
            print("Invalid operation. Please enter one of the following: save, retrieve, update, delete, exit.")
