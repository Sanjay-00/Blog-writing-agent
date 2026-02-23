import streamlit as st
from backend import app
from typing import Dict, Any

st.set_page_config(page_title="Blog Agent", layout="centered")

st.title("📝 Blog Writing Agent")

# ------------------------
# Input
# ------------------------
topic = st.text_input("Enter blog topic")
generate = st.button("Generate")

# ------------------------
# Placeholders
# ------------------------
node_status = st.empty()
output_area = st.empty()

# ------------------------
# Run Graph
# ------------------------
if generate and topic.strip():

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": "",
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "final": "",
    }

    with st.spinner("Running agent..."):

        try:
            for step in app.stream(inputs, stream_mode="updates"):

                # step usually looks like:
                # {"router": {...}}
                if isinstance(step, dict) and len(step) == 1:
                    node_name = list(step.keys())[0]
                    node_status.info(f"⚙️ Executing node: **{node_name}**")

            # After streaming finishes, get final state
            result = app.invoke(inputs)

            final_md = result.get("final", "")

            if final_md:
                node_status.success("✅ Completed")
                output_area.markdown(final_md)
            else:
                output_area.warning("No final output generated.")

        except Exception:
            # Fallback if stream not supported
            result = app.invoke(inputs)
            final_md = result.get("final", "")
            output_area.markdown(final_md)