from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from openai import OpenAI

from agents.risk_agent import risk_agent
from agents.return_agent import return_agent
from agents.suitability_agent import suitability_agent
from agents.macro_agent import macro_agent
from agents.aggregator_agent import aggregator_agent
from agents.utils import find_closest_fund


def run_parallel_analysis(fund_name: str):
    # Call aggregator_agent with the user's raw input for exact match
    final_decision = aggregator_agent(fund_name)
    if final_decision == "No match found":
        return None, final_decision

    # Only run LLM agents if a match is found
    agent_outputs = {
        "Risk Analysis": risk_agent(fund_name),
        "Return Analysis": return_agent(fund_name),
        "Suitability Analysis": suitability_agent(fund_name),
        "Macro Outlook": macro_agent(fund_name),
    }
    return agent_outputs, final_decision
