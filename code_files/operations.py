import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embeddings from CSV
def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    df['Embeddings'] = df['Embeddings'].apply(eval).apply(np.array)
    return df

# Save the DataFrame to CSV
def save_csv(df, csv_file):
    df.to_csv(csv_file, index=False)

# Encode text using SentenceTransformer
def encode_text(text, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embedding = model.encode([text], convert_to_tensor=True).cpu().numpy()[0]
    return embedding

# Find top K similar entries
def find_top_k_similar(query_embedding, df, k=5):
    embeddings = np.vstack(df['Embeddings'].values)
    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(query_embedding)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(query_embedding, k)
    return df.iloc[I[0]]

# Update an entry in the DataFrame
def update_entry(df, index, new_text, new_embedding):
    df.at[index, 'Text'] = new_text
    df.at[index, 'Embeddings'] = new_embedding
    return df

# Remove an entry from the DataFrame
def remove_entry(df, index):
    df = df.drop(index).reset_index(drop=True)
    return df


def save_entry(text, csv_file):
    df = load_csv(csv_file)
    new_embedding = encode_text(text)
    new_entry = {'Text': text, 'Embeddings': new_embedding}
    df = df.append(new_entry, ignore_index=True)
    save_csv(df, csv_file)
    print("Entry saved to CSV.")

def retrieve_entries(query, csv_file, k=5):
    df = load_csv(csv_file)
    query_embedding = encode_text(query)
    results = find_top_k_similar(query_embedding, df, k)
    print("Retrieved entries:")
    print(results[['Text']])
    return results

def update_entry_in_csv(index, new_text, csv_file):
    df = load_csv(csv_file)
    new_embedding = encode_text(new_text)
    df = update_entry(df, index, new_text, new_embedding)
    save_csv(df, csv_file)
    print("Entry updated in CSV.")

def remove_entry_from_csv(query, csv_file, k=5):
    df = load_csv(csv_file)
    query_embedding = encode_text(query)
    results = find_top_k_similar(query_embedding, df, k)
    print("Top similar entries for removal:")
    print(results[['Text']])
    index_to_remove = int(input("Enter the index of the entry to remove: "))
    df = remove_entry(df, index_to_remove)
    save_csv(df, csv_file)
    print("Entry removed from CSV.")
