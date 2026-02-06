def macro_agent(llm):
    prompt = PromptTemplate(
        input_variables=["fund"],
        template="""
        You are a macro-economic strategist.

        Analyze how current market and economic conditions
        impact the mutual fund: {fund}.
        """
    )
    return prompt | llm

from openai import OpenAI
from agents.utils import find_closest_fund

def macro_agent(fund_name):
    openai_client = OpenAI()
    fund, score = find_closest_fund(fund_name, openai_client)
    context = f"Fund Name: {fund['name']}\nMacro: {fund.get('macro', 'N/A')}\n"
    prompt = (
        f"You are a mutual fund macro analyst. Given the following fund and its macroeconomic context:\n{context}\n"
        "Provide a brief macroeconomic outlook for this fund."
    )
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a mutual fund macro analyst."},
                  {"role": "user", "content": prompt}]
    )
    return {"fund": fund['name'], "macro_analysis": completion.choices[0].message.content}
