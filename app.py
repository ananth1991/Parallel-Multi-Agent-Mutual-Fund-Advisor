import streamlit as st
from orchestrator import run_parallel_analysis

st.title("📊 Parallel Multi-Agent Mutual Fund Advisor")

fund = st.text_input(
    "Enter Mutual Fund Name",
    placeholder="e.g. Alpha Growth Fund"
)

if st.button("Analyze Fund"):
    if not fund.strip():
        st.warning("Please enter a mutual fund name.")
    else:
        with st.spinner("Running Orchestrator analysis..."):
            agent_results, final_decision = run_parallel_analysis(fund)

        # If fund not found, display error message
        if agent_results is None:
            st.error(final_decision)
        else:
            # Display agent results as accordions
            st.subheader("🔍 Agent Insights")
            
            for agent_name, output in agent_results.items():
                with st.expander(f"📋 {agent_name}"):
                    st.write(output.content)

            st.subheader("📌 Final Recommendation")
            st.success(final_decision)
