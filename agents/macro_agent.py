from langchain_core.prompts import PromptTemplate

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
