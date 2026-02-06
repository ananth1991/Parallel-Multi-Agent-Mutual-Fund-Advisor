import numpy as np
from openai import OpenAI

def load_fund_data():
    funds_meta = np.load("agents/funds_meta.npy", allow_pickle=True)
    fund_vectors = np.load("agents/funds_vectors.npy")
    return funds_meta, fund_vectors

def embed_query(query, openai_client):
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_closest_fund(query, openai_client):
    funds_meta, fund_vectors = load_fund_data()
    query_vec = embed_query(query, openai_client)
    similarities = np.array([cosine_similarity(query_vec, v) for v in fund_vectors])
    idx = np.argmax(similarities)
    return funds_meta[idx], similarities[idx]
