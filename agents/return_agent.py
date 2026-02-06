from langchain_core.prompts import PromptTemplate

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
