from openai import OpenAI
from agents.utils import find_closest_fund

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

def unified_agent(query):
    openai_client = OpenAI()
    fund, score = find_closest_fund(query, openai_client)
    context = (
        f"Name: {fund['name']}, Category: {fund['category']}, Risk: {fund['risk']}, "
        f"Returns: {fund['returns']}, Suitability: {fund['suitability']}, Macro: {fund['macro']}"
    )
    results = {}
    for agent in ["macro", "return", "risk", "suitability", "aggregator"]:
        results[agent] = run_agent(agent, context, openai_client)
    return results
    return completion.choices[0].message.content

def unified_agent(query):
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    vectors, funds_meta = load_fund_vectors()
    if vectors is None or funds_meta is None:
        return {"error": "fund vectors or metadata not found. Please run dummy_db_setup.py first."}
    query_vec = embed_query(query, openai_client)
    # Compute cosine similarity with all funds
    sims = np.array([cosine_similarity(query_vec, v) for v in vectors])
    top_k = 3
    top_indices = np.argsort(sims)[-top_k:][::-1]
    retrieved = [funds_meta[i] for i in top_indices]
    fund_context = "\n".join([
        f"Name: {fund['name']}, Category: {fund['category']}, Risk: {fund['risk']}, Returns: {fund['returns']}, Suitability: {fund['suitability']}, Macro: {fund['macro']}"
        for fund in retrieved
    ])
    results = {}
    for agent in ["macro", "return", "risk", "suitability", "aggregator"]:
        results[agent] = run_agent(agent, fund_context, openai_client)
    return results
