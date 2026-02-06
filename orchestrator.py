from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from openai import OpenAI

from agents.risk_agent import risk_agent
from agents.return_agent import return_agent
from agents.suitability_agent import suitability_agent
from agents.macro_agent import macro_agent
from agents.aggregator_agent import aggregator_agent, check_fund_exists


def run_parallel_analysis(fund_name: str):
    # First, check if fund exists in the local database
    openai_client = OpenAI()
    fund_exists, matched_fund = check_fund_exists(fund_name, openai_client)
    
    if not fund_exists:
        return None, f"No data found for '{fund_name}' in the local database. Available funds: Alpha Growth Fund, Beta Income Fund, Gamma Balanced Fund."
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    parallel_agents = RunnableParallel({
        "Risk Analysis": risk_agent(llm),
        "Return Analysis": return_agent(llm),
        "Suitability Analysis": suitability_agent(llm),
        "Macro Outlook": macro_agent(llm),
    })

    agent_outputs = parallel_agents.invoke({"fund": fund_name})

    final_runnable = aggregator_agent(fund_name)
    # Handle error if aggregator_agent returns a string
    if isinstance(final_runnable, str):
        return agent_outputs, f"Aggregator agent error: {final_runnable}"
    final_decision = final_runnable({
        "inputs": agent_outputs
    })
    return agent_outputs, final_decision
