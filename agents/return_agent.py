def return_agent(llm):
    prompt = PromptTemplate(
        input_variables=["fund"],
        template="""
        You are a mutual fund performance analyst.

        Analyze the return characteristics of the mutual fund: {fund}

        Focus on consistency, CAGR quality, and risk-adjusted returns.
        """
    )
    return prompt | llm

from openai import OpenAI
from agents.utils import find_closest_fund

def return_agent(fund_name):
    openai_client = OpenAI()
    fund, score = find_closest_fund(fund_name, openai_client)
    context = f"Fund Name: {fund['name']}\nReturns: {fund.get('returns', 'N/A')}\n"
    prompt = (
        f"You are a mutual fund returns analyst. Given the following fund and its returns data:\n{context}\n"
        "Provide a brief analysis of the fund's return profile."
    )
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a mutual fund returns analyst."},
                  {"role": "user", "content": prompt}]
    )
    return {"fund": fund['name'], "return_analysis": completion.choices[0].message.content}
