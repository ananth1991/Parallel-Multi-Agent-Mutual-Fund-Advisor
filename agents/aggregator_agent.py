import numpy as np
import os
from dotenv import load_dotenv

def load_fund_vectors():
    try:
        vectors = np.load("agents/funds_vectors.npy")
        funds_meta = np.load("agents/funds_meta.npy", allow_pickle=True)
        return vectors, funds_meta
    except Exception as e:
        print(f"Error loading fund vectors or metadata: {e}")
        return None, None

def aggregator_agent(query):
    load_dotenv()
    vectors, funds_meta = load_fund_vectors()
    if vectors is None or funds_meta is None:
        return "Error: fund vectors or metadata not found. Please run dummy_db_setup.py first."
    # Exact match (case-insensitive)
    query_lower = query.strip().lower()
    for fund in funds_meta:
        if fund.get("name", "").strip().lower() == query_lower:
            fund_name = fund.get("name", "Unknown Fund")
            fund_recommendation = fund.get("recommendation", "No recommendation available")
            return f"Recommendation for {fund_name}: {fund_recommendation}"
    return "No match found"