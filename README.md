# Parallel Multi-Agent Mutual Fund Advisor
### Built using LangChain & Streamlit

## Overview
This application is a **production-ready Parallel Multi-Agent System** designed to
analyze mutual funds from multiple independent perspectives and generate
an explainable investment recommendation.

The system leverages **LangChain's Parallel Execution pattern** to ensure:
- Independent reasoning
- Faster analysis
- Reduced bias
- Scalable architecture

---

## Business Problem
Mutual fund evaluation typically involves multiple dimensions:
- Risk
- Returns
- Investor suitability
- Market & macro environment

These dimensions are often analyzed in isolation or manually,
leading to inconsistent and delayed decisions.

---

## Solution
A **Parallel Multi-Agent Architecture** where:
- Each agent independently evaluates a mutual fund
- All agents execute simultaneously
- A final aggregator agent synthesizes the results
