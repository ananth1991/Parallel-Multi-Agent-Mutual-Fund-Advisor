from langchain_core.prompts import PromptTemplate

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
