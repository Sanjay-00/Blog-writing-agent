import streamlit as st
from backend import app
from typing import Dict, Any
import time

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="Blog Agent",
    layout="wide"
)

# ------------------------
# Center Layout
# ------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    st.title("📝 Blog Writing Agent")

    # ------------------------
    # Input
    # ------------------------
    topic = st.text_input("Enter blog topic")
    generate = st.button("Generate", use_container_width=True)

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
            "sections": [],
            "merged_md": "",
            "final": "",
        }

        final_state = {}
        step_number = 0
        current_node = None

        with st.spinner("Running agent..."):

            # Use updates mode to detect nodes
            for update in app.stream(inputs, stream_mode="updates"):

                if isinstance(update, dict) and len(update) == 1:

                    node_name = list(update.keys())[0]
                    node_output = list(update.values())[0]

                    # Track state progressively
                    if isinstance(node_output, dict):
                        final_state.update(node_output)

                    # Show step only when node changes
                    if node_name != current_node:
                        step_number += 1
                        current_node = node_name
                        node_status.info(
                            f"⚙️ Step {step_number} — Executing: **{node_name}**"
                        )

            node_status.success("✅ Blog generation completed")

        # ------------------------
        # ChatGPT-style Streaming Output
        # ------------------------
        final_md = final_state.get("final", "")

        if final_md:
            st.markdown("---")

            stream_placeholder = output_area.empty()
            streamed_text = ""

            # Stream character by character (preserves formatting)
            for char in final_md:
                streamed_text += char
                stream_placeholder.markdown(streamed_text)
                time.sleep(0.002)  # adjust speed here

        else:
            output_area.warning("No final output generated.")

        st.markdown("### ⬇️ Download")

        st.download_button(
            label="Download Blog as Markdown",
            data=final_md.encode("utf-8"),
            file_name="blog.md",
            mime="text/markdown",
            use_container_width=True
        )

    elif generate:
        st.warning("Please enter a topic.")