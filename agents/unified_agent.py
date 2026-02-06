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

def embed_query(query, openai_client):
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def run_agent(agent_type, fund, openai_client):
    if agent_type == "macro":
        prompt = f"You are a macro-economic strategist. Analyze how current market and economic conditions impact the mutual fund: {fund}."
    elif agent_type == "return":
        prompt = f"You are a mutual fund performance analyst. Analyze the return characteristics of the mutual fund: {fund}. Focus on consistency, CAGR quality, and risk-adjusted returns."
    elif agent_type == "risk":
        prompt = f"You are a professional mutual fund risk analyst. Analyze the risk profile of the mutual fund: {fund}. Consider volatility, drawdowns, and downside risk. Provide a concise investor-friendly assessment."
    elif agent_type == "suitability":
        prompt = f"You are an investment suitability advisor. Evaluate which investor profiles are suitable for the mutual fund: {fund}. Consider risk appetite and investment horizon."
    elif agent_type == "aggregator":
        prompt = f"You are a senior mutual fund advisor. Below are analyses of relevant funds: {fund}. Synthesize them and give a final recommendation (Buy / Hold / Avoid) with reasoning."
    else:
        return "Unknown agent type."
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    return completion.choices[0].message.content

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    index, funds_meta = load_faiss_index()
    if index is None or funds_meta is None:
        return {"error": "FAISS index or metadata not found. Please run dummy_db_setup.py first."}
    query_vec = embed_query(query, openai_client)
    D, I = index.search(np.expand_dims(query_vec, axis=0), k=3)
    retrieved = [funds_meta[i] for i in I[0]]
    fund_context = "\n".join([
        f"Name: {fund['name']}, Category: {fund['category']}, Risk: {fund['risk']}, Returns: {fund['returns']}, Suitability: {fund['suitability']}, Macro: {fund['macro']}"
        for fund in retrieved
    ])
    results = {}
    for agent in ["macro", "return", "risk", "suitability", "aggregator"]:
        results[agent] = run_agent(agent, fund_context, openai_client)
    return results
