def risk_agent(llm):
    prompt = PromptTemplate(
        input_variables=["fund"],
        template="""
        You are a professional mutual fund risk analyst.

        Analyze the risk profile of the mutual fund: {fund}

        Consider volatility, drawdowns, and downside risk.
        Provide a concise investor-friendly assessment.
        """
    )
    return prompt | llm

from openai import OpenAI
from agents.utils import find_closest_fund

def risk_agent(fund_name):
    openai_client = OpenAI()
    fund, score = find_closest_fund(fund_name, openai_client)
    context = f"Fund Name: {fund['name']}\nRisk: {fund.get('risk', 'N/A')}\n"
    prompt = (
        f"You are a mutual fund risk analyst. Given the following fund and its risk data:\n{context}\n"
        "Provide a brief analysis of the fund's risk profile."
    )
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a mutual fund risk analyst."},
                  {"role": "user", "content": prompt}]
    )
    return {"fund": fund['name'], "risk_analysis": completion.choices[0].message.content}
