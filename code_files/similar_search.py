import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_embeddings(csv_file):
    df = pd.read_csv(csv_file)
    embeddings = df["Embeddings"].apply(eval).apply(np.array).tolist()
    texts = df["Text"].tolist()
    return texts, np.ascontiguousarray(embeddings, dtype=np.float32)

def encode_query(query, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], convert_to_tensor=True)
    return np.ascontiguousarray(query_embedding.cpu().numpy(), dtype=np.float32)

def find_top_k_similar_texts(query_embedding, texts, embeddings, k=3, metric='euclidean'):
    if metric == 'cosine':
        faiss.normalize_L2(embeddings)
        faiss.normalize_L2(query_embedding)
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product (cosine similarity)
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])  # Euclidean distance
    
    index.add(embeddings)
    D, I = index.search(query_embedding, k)
    top_k_texts = [texts[i] for i in I[0]]
    return top_k_texts, D[0]


def mmr(doc_embeddings, query_embedding, texts, top_k=5, lambda_param=0.5):
    """
    MMR implementation for document retrieval.
    """
    # Normalize embeddings
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Similarity scores
    sim_scores = np.dot(doc_embeddings, query_embedding.T).flatten()
    
    selected = []
    selected_scores = []
    
    while len(selected) < top_k:
        if not selected:
            # Select the first document with the highest similarity
            selected_idx = np.argmax(sim_scores)
        else:
            # Calculate MMR for remaining documents
            mmr_scores = []
            for idx in range(len(doc_embeddings)):
                if idx not in selected:
                    diversity = max([np.dot(doc_embeddings[idx], doc_embeddings[sel_idx]) for sel_idx in selected])
                    mmr_score = lambda_param * sim_scores[idx] - (1 - lambda_param) * diversity
                    mmr_scores.append((mmr_score, idx))
            selected_idx = max(mmr_scores)[1]
        
        selected.append(selected_idx)
        selected_scores.append(sim_scores[selected_idx])
    
    return [texts[i] for i in selected], selected_scores

if __name__ == "__main__":
    csv_file = 'extracted_html_files/saivineethkumar.github.io/chunks_and_embeddings.csv'
    texts, embeddings = load_embeddings(csv_file)

    query = "Hello tell me something about me"
    query_embedding = encode_query(query)

    # Choose the distance metric 'cosine' or 'euclidean'
    # metric = 'cosine'  
    # metric = 'euclidean'
    metric = 'mmr'
    # print(metric)


    if metric in ['cosine', 'euclidean']: 
        top_k_texts, distances = find_top_k_similar_texts(query_embedding, texts, embeddings, k=3, metric=metric)
    elif metric == 'mmr':
        top_k_texts, scores = mmr(embeddings, query_embedding, texts, top_k=3)

    print("Top 3 similar texts:")
    for i, (text, distance) in enumerate(zip(top_k_texts, distances)):
        print(f"Rank {i + 1}: {text}")
        print(f"Distance: {distance}")
