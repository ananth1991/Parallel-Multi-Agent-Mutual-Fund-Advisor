import numpy as np
import faiss
from openai import OpenAI
import os
from dotenv import load_dotenv

def load_faiss_index():
    try:
        index = faiss.read_index("agents/funds.index")
        funds_meta = np.load("agents/funds_meta.npy", allow_pickle=True)
        return index, funds_meta
    except Exception as e:
        print(f"Error loading FAISS index or metadata: {e}")
        return None, None

def check_fund_exists(query, openai_client):
    """Check if a fund exists in the local database based on similarity threshold."""
    index, funds_meta = load_faiss_index()
    if index is None or funds_meta is None:
        return False, None
    
    query_vec = embed_query(query, openai_client)
    D, I = index.search(np.expand_dims(query_vec, axis=0), k=1)  # Top 1 match
    
    # Set very strict distance threshold for relevance (FAISS L2 distance)
    # Only accept very close semantic matches
    DISTANCE_THRESHOLD = 0.3
    best_distance = D[0][0]  # Distance of the best match
    
    if best_distance <= DISTANCE_THRESHOLD:
        # Fund found, return the fund data
        matched_fund = funds_meta[I[0][0]]
        return True, matched_fund
    else:
        return False, None

def embed_query(query, openai_client):
    # Ensure input is a string
    if not isinstance(query, str):
        query = str(query)
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def aggregator_agent(query):
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    index, funds_meta = load_faiss_index()
    if index is None or funds_meta is None:
        return "Error: FAISS index or metadata not found. Please run dummy_db_setup.py first."
    query_vec = embed_query(query, openai_client)
    D, I = index.search(np.expand_dims(query_vec, axis=0), k=1)  # Top 1 match
    
    retrieved_fund = funds_meta[I[0][0]]
    
    def aggregator_runnable(inputs):
        agent_outputs = inputs.get("inputs", {})
        
        # Get recommendation from fund data - return only the recommendation
        fund_recommendation = retrieved_fund.get("recommendation", "No recommendation available")
        
        return fund_recommendation
    return aggregator_runnable
