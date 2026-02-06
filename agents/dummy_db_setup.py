import numpy as np
import faiss
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Dummy fund data
funds = [
    {
        "name": "Alpha Growth Fund",
        "category": "Equity",
        "risk": "High",
        "returns": 12.5,
        "suitability": "Aggressive",
        "macro": "Positive macro outlook",
        "recommendation": "Strong Buy for next 5 years"
    },
    {
        "name": "Beta Income Fund",
        "category": "Debt",
        "risk": "Low",
        "returns": 7.2,
        "suitability": "Conservative",
        "macro": "Stable macro environment",
        "recommendation": "Buy for next 5 years with only 8% return expectation"
    },
    {
        "name": "Gamma Balanced Fund",
        "category": "Hybrid",
        "risk": "Medium",
        "returns": 9.8,
        "suitability": "Moderate",
        "macro": "Neutral macro factors",
        "recommendation": "Sell/Don't buy as it is continuously under performing the benchmark in last 5 years"
    }
]

# Use OpenAI to embed fund descriptions
def get_fund_vector(fund, openai_client):
    desc = f"{fund['name']} {fund['category']} {fund['risk']} {fund['returns']} {fund['suitability']} {fund['macro']}"
    # Ensure input is a string
    if not isinstance(desc, str):
        desc = str(desc)
    response = openai_client.embeddings.create(
        input=desc,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def create_faiss_index():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    dim = 1536  # Embedding dimension for ada-002
    index = faiss.IndexFlatL2(dim)
    vectors = []
    for fund in funds:
        vec = get_fund_vector(fund, openai_client)
        vectors.append(vec)
    vectors_np = np.vstack(vectors)
    index.add(vectors_np)
    # Ensure agents/ directory exists
    os.makedirs("agents", exist_ok=True)
    faiss.write_index(index, "agents/funds.index")
    np.save("agents/funds_meta.npy", funds)
    print("FAISS index and metadata saved in agents/.")

if __name__ == "__main__":
    create_faiss_index()
