def suitability_agent(llm):
    prompt = PromptTemplate(
        input_variables=["fund"],
        template="""
        You are an investment suitability advisor.

        Evaluate which investor profiles are suitable for the mutual fund: {fund}

        Consider risk appetite and investment horizon.
        """
    )
    return prompt | llm

from openai import OpenAI
from agents.utils import find_closest_fund

def suitability_agent(fund_name):
    openai_client = OpenAI()
    fund, score = find_closest_fund(fund_name, openai_client)
    context = f"Fund Name: {fund['name']}\nSuitability: {fund.get('suitability', 'N/A')}\n"
    prompt = (
        f"You are a mutual fund suitability analyst. Given the following fund and its suitability information:\n{context}\n"
        "Provide a brief analysis of which investor profiles this fund is suitable for."
    )
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a mutual fund suitability analyst."},
                  {"role": "user", "content": prompt}]
    )
    return {"fund": fund['name'], "suitability_analysis": completion.choices[0].message.content}
