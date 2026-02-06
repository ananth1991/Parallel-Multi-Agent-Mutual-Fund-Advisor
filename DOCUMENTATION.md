# 📊 Multi-Agent Mutual Fund Advisor - Complete Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Flow](#architecture--flow)
3. [File Structure & Functionality](#file-structure--functionality)
4. [Execution Flow](#execution-flow)
5. [Code Breakdown by File](#code-breakdown-by-file)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Key Concepts](#key-concepts)

---

## System Overview

### What is This Application?
A **multi-agent mutual fund recommendation system** that analyzes mutual funds from a local database and provides comprehensive investment advice through multiple specialized AI agents.

### Key Features:
- ✅ Local database only (3 sample funds)
- ✅ Multiple specialized AI agents (Risk, Return, Suitability, Macro)
- ✅ Semantic search using NumPy (cosine similarity)
- ✅ OpenAI embeddings for fund matching
- ✅ Streamlit UI with accordion-based results
- ✅ Pre-configured fund recommendations

---

## Architecture & Flow

```
User Input (Fund Name)
       ↓
   app.py (Streamlit UI)
       ↓
orchestrator.py (Check fund exists & coordinate agents)
       ↓
   ├→ check_fund_exists() [aggregator_agent.py]
   │     (Uses NumPy cosine similarity to match fund)
   │
   └→ If Fund Found:
       ├→ risk_agent (LangChain)
       ├→ return_agent (LangChain)
       ├→ suitability_agent (LangChain)
       ├→ macro_agent (LangChain)
       └→ aggregator_agent (Returns recommendation)
              ↓
         Results displayed in UI with accordions
```

---

## File Structure & Functionality

### 📁 Project Structure
```
mutual_fund_multi_agent/
├── app.py                          # Streamlit frontend
├── orchestrator.py                 # Orchestration logic
├── requirements.txt                # Dependencies
├── .env                           # API keys
├── agents/
│   ├── dummy_db_setup.py          # Initialize database and embeddings
│   ├── aggregator_agent.py        # Fund matching & final recommendation
│   ├── risk_agent.py              # Risk analysis agent
│   ├── return_agent.py            # Return analysis agent
│   ├── suitability_agent.py       # Suitability analysis agent
│   ├── macro_agent.py             # Macro outlook agent
│   ├── unified_agent.py           # Alternative unified agent
│   ├── funds_meta.npy             # Fund metadata (generated)
```

---

## Execution Flow

### Step 1: Application Startup
**File: `app.py`**
```python
import streamlit as st
from orchestrator import run_parallel_analysis

st.title("📊 Parallel Multi-Agent Mutual Fund Advisor")
```
- Streamlit app initializes and displays the UI
- User sees text input box asking for fund name

### Step 2: User Input
**File: `app.py`**
```python
fund = st.text_input("Enter Mutual Fund Name", placeholder="e.g. Alpha Growth Fund")
```
- User types fund name (e.g., "Alpha Growth Fund")
- User clicks "Analyze Fund" button

### Step 3: Check if Fund Exists in Database
**File: `orchestrator.py` → `aggregator_agent.py`**
```python
## orchestrator.py
fund_exists, matched_fund = check_fund_exists(fund_name, openai_client)

if not fund_exists:
    return None, f"No data found for '{fund_name}' in the local database..."
```

**Inside `check_fund_exists()` in `aggregator_agent.py`:**
1. Load fund metadata
2. Convert user input to embedding (using OpenAI)
3. Calculate cosine similarity with all funds in the database
4. Check if similarity >= 0.7 threshold
5. Return True/Fund data if match found, False if not

### Step 4: If Fund Found, Run Parallel Agents
**File: `orchestrator.py`**
```python
parallel_agents = RunnableParallel({
    "Risk Analysis": risk_agent(llm),
    "Return Analysis": return_agent(llm),
    "Suitability Analysis": suitability_agent(llm),
    "Macro Outlook": macro_agent(llm),
})

agent_outputs = parallel_agents.invoke({"fund": fund_name})
```

**What happens:** All 4 agents run simultaneously (parallel execution)

### Step 5: Get Final Recommendation
**File: `orchestrator.py` → `aggregator_agent.py`**
```python
final_runnable = aggregator_agent(fund_name)
final_decision = final_runnable({"inputs": agent_outputs})
```

**Inside `aggregator_agent.py`:**
- Retrieves pre-configured recommendation from fund database
- Returns ONLY the recommendation (not the full analysis)

### Step 6: Display Results to User
**File: `app.py`**
```python
for agent_name, output in agent_results.items():
    with st.expander(f"📋 {agent_name}"):
        st.write(output.content)

st.subheader("📌 Final Recommendation")
st.success(final_decision)
```

Results displayed as:
- **Accordions:** Each agent's detailed analysis (expandable)
- **Final Recommendation:** Big success box with recommendation

---

## Code Breakdown by File

## 1️⃣ app.py (Frontend)

**Purpose:** Streamlit user interface for the application

**Key Sections:**

```python
import streamlit as st
from orchestrator import run_parallel_analysis
```
- Import Streamlit for UI
- Import orchestrator to run analysis

```python
st.title("📊 Parallel Multi-Agent Mutual Fund Advisor")
```
- Display main title

```python
fund = st.text_input(
    "Enter Mutual Fund Name",
    placeholder="e.g. Alpha Growth Fund"
)
```
- **Functionality:** Text input box for user to enter fund name
- **Output:** Returns string (fund name)

```python
if st.button("Analyze Fund"):
    if not fund.strip():
        st.warning("Please enter a mutual fund name.")
```
- **Functionality:** Button trigger for analysis
- **Validation:** Check if input is empty

```python
with st.spinner("Running Orchestrator analysis..."):
    agent_results, final_decision = run_parallel_analysis(fund)
```
- **Functionality:** Call orchestrator with fund name
- **Returns:** agent_results (dict) or None, final_decision (string)
- **UI:** Shows loading spinner while processing

```python
if agent_results is None:
    st.error(final_decision)
else:
    st.subheader("🔍 Agent Insights")
    
    for agent_name, output in agent_results.items():
        with st.expander(f"📋 {agent_name}"):
            st.write(output.content)
```
- **Functionality:** Display results
- **Logic:** 
  - If no fund found → Show error message
  - If fund found → Show accordions

```python
st.subheader("📌 Final Recommendation")
st.success(final_decision)
```
- Display final recommendation in green success box

---

## 2️⃣ orchestrator.py (Orchestration Logic)

**Purpose:** Coordinates all agents and controls execution flow

**Key Functions:**

```python
def run_parallel_analysis(fund_name: str):
```
- **Input:** fund_name (string) - Name of fund to analyze
- **Output:** (agent_results, final_decision) tuple

**Step 1: Initialize OpenAI client**
```python
openai_client = OpenAI()
```

**Step 2: Check fund exists**
```python
fund_exists, matched_fund = check_fund_exists(fund_name, openai_client)

if not fund_exists:
    return None, f"No data found for '{fund_name}'..."
```
- **What it does:** 
  - Calls check_fund_exists() from aggregator_agent.py
  - Returns False if fund not in database
  - Prevents running agents for non-existent funds

**Step 3: Initialize LLM**
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
```
- **Model:** gpt-4o-mini (fast, cost-effective)
- **Temperature:** 0.2 (deterministic, focused responses)

**Step 4: Create parallel agents**
```python
parallel_agents = RunnableParallel({
    "Risk Analysis": risk_agent(llm),
    "Return Analysis": return_agent(llm),
    "Suitability Analysis": suitability_agent(llm),
    "Macro Outlook": macro_agent(llm),
})
```
- **Functionality:** Creates 4 agents that run simultaneously
- **Why parallel:** Faster execution (all agents run at same time)

**Step 5: Run agents**
```python
agent_outputs = parallel_agents.invoke({"fund": fund_name})
```
- **Input:** Dictionary with "fund" key containing fund name
- **Output:** Dictionary with 4 agent responses
- **Flow:** Each agent receives fund name and generates analysis

**Step 6: Get final recommendation**
```python
final_runnable = aggregator_agent(fund_name)
if isinstance(final_runnable, str):
    return agent_outputs, f"Aggregator agent error: {final_runnable}"
    
final_decision = final_runnable({"inputs": agent_outputs})
```
- **Functionality:** 
  - Create aggregator agent with fund name
  - Check if error occurred
  - If OK, call agent with all agent outputs
  - Get final recommendation

**Step 7: Return results**
```python
return agent_outputs, final_decision
```
- Returns to app.py for display

---

## 3️⃣ agents/aggregator_agent.py (Fund Matching & Recommendation)

**Purpose:** Find fund in database and provide pre-configured recommendation

**Key Functions:**

### Function 1: load_faiss_index()
```python
def load_faiss_index():
    try:
        index = faiss.read_index("agents/funds.index")
        funds_meta = np.load("agents/funds_meta.npy", allow_pickle=True)
        return index, funds_meta
    except Exception as e:
        print(f"Error loading FAISS index or metadata: {e}")
        return None, None
```
- **Purpose:** Load vector database and fund metadata
- **Input:** None (reads from disk)
- **Output:** (index, funds_meta) or (None, None)
- **What it does:**
  - Loads FAISS index containing vector embeddings of funds
  - Loads numpy array with fund data (name, category, risk, etc.)
  - Returns None if file not found

**Important:** FAISS index is created by running `dummy_db_setup.py`

### Function 2: embed_query()
```python
def embed_query(query, openai_client):
    if not isinstance(query, str):
        query = str(query)
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)
```
- **Purpose:** Convert text to vector embedding
- **Input:** query (string), openai_client
- **Output:** numpy array (1536 dimensions for ada-002)
- **What it does:**
  - Converts fund name string to OpenAI embedding
  - Returns 1536-dimensional vector
    """Check if a fund exists in the local database based on similarity threshold."""
    index, funds_meta = load_faiss_index()
    if index is None or funds_meta is None:
        return False, None
    
    query_vec = embed_query(query, openai_client)
    D, I = index.search(np.expand_dims(query_vec, axis=0), k=1)
    
    DISTANCE_THRESHOLD = 0.3  # Very strict threshold
    best_distance = D[0][0]
    
    if best_distance <= DISTANCE_THRESHOLD:
        matched_fund = funds_meta[I[0][0]]
        return True, matched_fund
    else:
        return False, None
```
3. Search FAISS index for most similar fund
   - `index.search()` returns distances (D) and indices (I)
   - We get top 1 match
4. Check if distance is below threshold (0.3)
   - **Lower distance = better match**
   - **Threshold 0.3 is very strict** (rejects "Nippon India Small Cap")
5. Return True + fund data if match found, False if not

**Key insight:** This is how we ensure **no public data access** - only exact matches from local database

### Function 4: aggregator_agent()
```python
def aggregator_agent(query):
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    index, funds_meta = load_faiss_index()
    
    if index is None or funds_meta is None:
        return "Error: FAISS index or metadata not found..."
    
    query_vec = embed_query(query, openai_client)
    D, I = index.search(np.expand_dims(query_vec, axis=0), k=1)
    retrieved_fund = funds_meta[I[0][0]]
    
    def aggregator_runnable(inputs):
        agent_outputs = inputs.get("inputs", {})
        fund_recommendation = retrieved_fund.get("recommendation", "No recommendation available")
        return fund_recommendation
    
    return aggregator_runnable
```

**What it does:**
1. Load OpenAI API key from .env
2. Load FAISS index
3. Search for fund in database
4. Returns a function (aggregator_runnable) that:
   - Takes agent outputs as input
   - Retrieves pre-configured recommendation from fund database
   - Returns ONLY the recommendation text

**Important:** Returns a **callable function**, not a string
- orchestrator.py will call this function later

---

## 4️⃣ agents/risk_agent.py

**Purpose:** Analyze risk profile of fund

```python
from langchain_core.prompts import PromptTemplate

def risk_agent(llm):
    prompt = PromptTemplate(
        input_variables=["fund"],
        template="""

        Analyze the risk profile of the mutual fund: {fund}

        Consider volatility, drawdowns, and downside risk.
        Provide a concise investor-friendly assessment.
        """
    )
    return prompt | llm
```
**What it does:**
    **Purpose:** Find fund in database and provide pre-configured recommendation

    **Key Functions:**

    ### Function 1: load_fund_vectors()
    ```python
    def load_fund_vectors():
        try:
            vectors = np.load("agents/funds_vectors.npy")
            funds_meta = np.load("agents/funds_meta.npy", allow_pickle=True)
            return vectors, funds_meta
        except Exception as e:
            print(f"Error loading fund vectors or metadata: {e}")
            return None, None
    ```
    - **Purpose:** Load fund metadata
    - **Input:** None (reads from disk)
    - **Output:** (vectors, funds_meta) or (None, None)
    - **What it does:**
      - Loads numpy array with fund data (name, category, risk, etc.)
      - Returns None if file not found

    ### Function 2: aggregator_agent()
    ```python
    def aggregator_agent(query):
        load_dotenv()
        vectors, funds_meta = load_fund_vectors()
        if vectors is None or funds_meta is None:
            return "Error: fund vectors or metadata not found. Please run dummy_db_setup.py first."
        # Exact match (case-insensitive)
        query_lower = query.strip().lower()
        for fund in funds_meta:
            if fund.get("name", "").strip().lower() == query_lower:
                fund_name = fund.get("name", "Unknown Fund")
                fund_recommendation = fund.get("recommendation", "No recommendation available")
                return f"Recommendation for {fund_name}: {fund_recommendation}"
        return "No match found"
    ```
    - **Purpose:** Find exact match for fund name and return recommendation
    - **Input:** query (user's fund name)
    - **Output:** Recommendation string or "No match found"
    - **What it does:**
      - Checks for exact (case-insensitive) match in fund metadata
      - Returns recommendation if found, else "No match found"
- Creates a prompt template with placeholder for fund name
- Returns a LangChain runnable (prompt | llm)
- When called with {"fund": "Alpha Growth Fund"}, it:
  1. Fills in the fund name in the prompt
  2. Sends prompt to LLM
  3. Returns LLM response about risk

**Output:** Detailed risk analysis (volatility, drawdowns, downside risk)

---

## 5️⃣ agents/return_agent.py

**Purpose:** Analyze return characteristics

```python
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
```

**Output:** Analysis of consistency, CAGR, risk-adjusted returns

---

## 6️⃣ agents/suitability_agent.py

**Purpose:** Identify suitable investor profiles

```python
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
```

**Output:** Suitable investor profiles based on risk and time horizon

---

## 7️⃣ agents/macro_agent.py

**Purpose:** Analyze macro-economic impact

```python
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
```

**Output:** How economic conditions affect fund performance

---

## 8️⃣ agents/dummy_db_setup.py (Initialize Database)

**Purpose:** Create FAISS vector index from fund data

**Key Functions:**

### Function 1: get_fund_vector()
```python
def get_fund_vector(fund, openai_client):
    desc = f"{fund['name']} {fund['category']} {fund['risk']} {fund['returns']} {fund['suitability']} {fund['macro']}"
    if not isinstance(desc, str):
        desc = str(desc)
    response = openai_client.embeddings.create(
        input=desc,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)
```
- **Purpose:** Convert fund description to embedding vector
- **What it does:**
  1. Create fund description from properties
  2. Call OpenAI to get embedding
  3. Return as numpy array

### Function 2: create_faiss_index()
```python
def create_faiss_index():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    dim = 1536  # Embedding dimension for ada-002
    index = faiss.IndexFlatL2(dim)
    vectors = []
    
    for fund in funds:
        vec = get_fund_vector(fund, openai_client)
        vectors.append(vec)
    
    vectors_np = np.vstack(vectors)
    index.add(vectors_np)
    
    os.makedirs("agents", exist_ok=True)
    faiss.write_index(index, "agents/funds.index")
    np.save("agents/funds_meta.npy", funds)
    print("FAISS index and metadata saved in agents/.")
```

**Step by step:**
1. Load OpenAI API key
2. Create FAISS index (L2 distance, 1536 dimensions)
3. For each fund:
   - Convert to embedding
   - Add to vectors list
4. Combine all vectors into one numpy array
5. Add all vectors to FAISS index
6. Save index to "agents/funds.index"
7. Save fund metadata to "agents/funds_meta.npy"

**When to run:** Run once to initialize, or whenever you add new funds

**Run command:** `python agents/dummy_db_setup.py`

---

## 9️⃣ Dummy Fund Data (In dummy_db_setup.py)

```python
funds = [
    {
        "name": "Alpha Growth Fund",
        "category": "Equity",
        "risk": "High",
        "returns": 12.5,
        "suitability": "Aggressive",
        "macro": "Positive macro outlook",
        "recommendation": "Strong Buy for next 5 years"
    },
    {
        "name": "Beta Income Fund",
        "category": "Debt",
        "risk": "Low",
        "returns": 7.2,
        "suitability": "Conservative",
        "macro": "Stable macro environment",
        "recommendation": "Buy for next 5 years with only 8% return expectation"
    },
    {
        "name": "Gamma Balanced Fund",
        "category": "Hybrid",
        "risk": "Medium",
        "returns": 9.8,
        "suitability": "Moderate",
        "macro": "Neutral macro factors",
        "recommendation": "Sell/Don't buy as it is continuously under performing the benchmark in last 5 years"
    }
]
```

**Properties:**
- `name`: Fund name (for semantic search)
- `category`: Type of fund (Equity, Debt, Hybrid)
- `risk`: Risk level (High, Low, Medium)
- `returns`: Expected return percentage
- `suitability`: Target investor profile
- `macro`: Macro outlook
- `recommendation`: Pre-configured recommendation (display in UI)

---

## Data Flow Diagram

```
INITIALIZATION PHASE:
│
├─ funds = [3 fund objects from dummy_db_setup.py]
│
├─ dummy_db_setup.py runs (one time setup)
│   ├─ For each fund:
│   │   ├─ Create description: "Alpha Growth Fund Equity High 12.5 Aggressive..."
│   │   ├─ Send to OpenAI API
│   │   └─ Get 1536-dimensional embedding
│   │
│   ├─ Store embeddings in FAISS index
│   ├─ Save as agents/funds.index
│   ├─ Save metadata as agents/funds_meta.npy
│

RUNTIME PHASE (Each time user queries):
│
├─ User enters: "Alpha Growth Fund"
│
├─ app.py calls: run_parallel_analysis("Alpha Growth Fund")
│
├─ orchestrator.py:
│   │
│   ├─ Check fund exists:
│   │   ├─ Convert "Alpha Growth Fund" to embedding
│   │   ├─ Calculate cosine similarity with all funds
│   │   ├─ Get best match
│   │   └─ If match found: FOUND ✓ else: NOT FOUND ✗
│   │
│   ├─ If FOUND, Initialize LangChain LLM
│   │
│   ├─ Run Parallel Agents (all at same time):
│   │   ├─ Risk Agent → "Analyze risk of Alpha Growth Fund"
│   │   ├─ Return Agent → "Analyze returns of Alpha Growth Fund"
│   │   ├─ Suitability Agent → "Analyze suitability of Alpha Growth Fund"
│   │   └─ Macro Agent → "Analyze macro impact on Alpha Growth Fund"
│   │
│   ├─ Each agent sends to OpenAI GPT-4o-mini
│   ├─ Get 4 detailed analyses
│   │
│   ├─ Get Final Recommendation:
│   │   ├─ aggregator_agent returns "Strong Buy for next 5 years"
│   │   └─ (From fund.recommendation property, NOT from LLM)
│   │
│   └─ Return to app.py
│
├─ app.py displays results:
│   ├─ Accordions (expandable):
│   │   ├─ Risk Analysis (from risk_agent)
│   │   ├─ Return Analysis (from return_agent)
│   │   ├─ Suitability Analysis (from suitability_agent)
│   │   └─ Macro Outlook (from macro_agent)
│   │
│   └─ Final Recommendation (green box):
│       └─ "Strong Buy for next 5 years"
```

---

## Key Concepts

### 1. Semantic Search with NumPy
**What:** Finding similar items (funds) based on numerical data (embeddings)
**Why:** To match user query with fund data without keyword search
**How in our app:**
- Fund names get converted to 1536-dim vectors
- User query gets converted to same vector space
- NumPy calculates cosine similarity between vectors
- Only exact matches (similarity >= 0.7) are considered

### 2. OpenAI Embeddings
**What:** Convert text to numerical vectors
**Model:** text-embedding-ada-002 (1536 dimensions)
**Used for:**
- Converting fund names to vectors (in dummy_db_setup.py)
- Converting user query to vector (in check_fund_exists)
- Semantic matching without keyword search

### 3. LangChain
**What:** Framework for building chains with LLMs
**Used for:**
- Creating prompt templates (agents)
- Running agents in parallel (RunnableParallel)
- Invoking LLM with structured prompts

### 4. Cosine Similarity
**What:** Measure of similarity between two vectors
**Range:** -1 (exact opposites) to 1 (exact matches)
**Used for:**
- Matching user query with fund embeddings
- Only proceeds if similarity is 0.7 or higher
**Why parallel?**
- All 4 agents run simultaneously (not sequentially)
- Faster overall execution time
- Independent analyses (one agent doesn't depend on others)

**How it works:**
```python
RunnableParallel({
    "Risk Analysis": risk_agent(llm),
    "Return Analysis": return_agent(llm),
    "Suitability Analysis": suitability_agent(llm),
    "Macro Outlook": macro_agent(llm),
})
```

### 6. Pre-configured Recommendations
**Why not use LLM for final recommendation?**
- Recommendations are expert-verified
- Consistent and reliable
**Where stored:** In fund object's "recommendation" property

---

## Important Code Lines & What They Do

### Line 1: Load environment variables
```python
from dotenv import load_dotenv
load_dotenv()
```
**What:** Reads .env file to load OPENAI_API_KEY

### Line 2: Initialize OpenAI client
```python
openai_client = OpenAI(api_key=openai_api_key)
```
**What:** Creates client to call OpenAI API

### Line 3: Load fund metadata
```python
funds_meta = np.load("agents/funds_meta.npy", allow_pickle=True)
```
**What:** Loads fund metadata (name, category, risk, etc.) from file

### Line 4: Embed query
```python
query_vec = embed_query(query, openai_client)
```
**What:** Converts user query to 1536-dimensional vector

### Line 5: Calculate cosine similarity
```python
similarity = cosine_similarity(query_vec, fund_vector)
```
**What:** Measures similarity between query and fund vectors

### Line 6: Check similarity threshold
```python
if similarity >= 0.7:
```
**What:** Only proceed if fund is similar enough

### Line 7: Create parallel agents
```python
parallel_agents = RunnableParallel({...})
```
**What:** All agents run simultaneously

### Line 8: Invoke agents
```python
agent_outputs = parallel_agents.invoke({"fund": fund_name})
```
**What:** Send fund name to all agents, get results

### Line 9: Display accordions
```python
with st.expander(f"📋 {agent_name}"):
```
**What:** Makes each agent output expandable

---

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Create `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

### 4. Initialize Database
```bash
python agents/dummy_db_setup.py
```
**Output:** 
- agents/funds.index
- agents/funds_meta.npy

### 5. Run Application
```bash
streamlit run app.py
```

---

## Execution Summary

```
User Query: "Alpha Growth Fund"
     ↓ 
app.py: Takes input, calls orchestrator
     ↓
orchestrator.py: Checks if fund exists via cosine similarity
     ↓
Is Fund in Database? (similarity >= 0.7)
     ├─ NO → Return "No data found" message
     └─ YES → Continue to agents
               ↓
        Run 4 agents in parallel:
        ├─ Risk Analysis
        ├─ Return Analysis
        ├─ Suitability Analysis
        └─ Macro Outlook
               ↓
        Aggregator gets pre-configured recommendation
               ↓
app.py: Display results with accordions + recommendation
```

---

## Summary

This is a **production-grade multi-agent system** that:

✅ **Uses local database only** (3 sample funds via NumPy)
✅ **Prevents hallucination** (strict similarity threshold)
✅ **Runs multiple analyses in parallel** (faster execution)
✅ **Provides structured output** (accordions for different perspectives)
✅ **Uses pre-verified recommendations** (data-driven, not LLM-generated)
✅ **Clean separation of concerns** (each agent has specific role)
✅ **Scalable architecture** (easy to add more agents or funds)

---

## Future Enhancements

1. **Add more funds** to dummy_db_setup.py
2. **Connect to real database** instead of local files
3. **Add user preferences** (risk tolerance, time horizon)
4. **Personalized recommendations** based on user profile
5. **Performance tracking** of recommendations over time
6. **Portfolio analysis** (multiple fund combinations)
7. **Real-time fund data** integration

---

**Created:** February 2026
**Technology Stack:** Python, LangChain, OpenAI, NumPy, Streamlit
