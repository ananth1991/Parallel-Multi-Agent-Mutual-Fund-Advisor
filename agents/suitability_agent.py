from langchain_core.prompts import PromptTemplate

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
