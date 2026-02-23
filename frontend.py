import streamlit as st
from backend import app
from typing import Dict, Any
from pathlib import Path
import time
import re

st.set_page_config(page_title="Blog Agent", layout="wide")

FILES_DIR = Path("files")
FILES_DIR.mkdir(exist_ok=True)

# Track selected blog in session
if "selected_blog" not in st.session_state:
    st.session_state.selected_blog = None

# ------------------------
# Layout (20% / 80%)
# ------------------------
left, right = st.columns([1, 4])

# ------------------------
# LEFT PANEL (Scrollable History)
# ------------------------
with left:
    st.markdown("## 📚 History")

    search = st.text_input("Search blog")

    blog_files = sorted(FILES_DIR.glob("*.md"), reverse=True)

    if search:
        blog_files = [
            f for f in blog_files
            if search.lower() in f.name.lower()
        ]

    # Scrollable container
    history_container = st.container(height=600)

    with history_container:
        for file in blog_files:
            if st.button(file.stem, use_container_width=True):
                st.session_state.selected_blog = file

# ------------------------
# RIGHT PANEL (Scrollable Blog Area)
# ------------------------
with right:

    st.title("📝 Blog Writing Agent")

    with st.form("blog_form"):
        topic = st.text_input("Enter blog topic")
        generate = st.form_submit_button("Generate")

    status_area = st.empty()
    blog_container = st.container(height=700)

    # ------------------------
    # Load Selected Blog
    # ------------------------
    if st.session_state.selected_blog and not generate:
        content = st.session_state.selected_blog.read_text(encoding="utf-8")
        with blog_container:
            st.markdown(content)

            st.download_button(
                "⬇️ Download",
                data=content.encode("utf-8"),
                file_name=st.session_state.selected_blog.name,
                mime="text/markdown",
                use_container_width=True
            )

    # ------------------------
    # Generate New Blog
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
        current_node = None

        with st.spinner("Generating blog..."):

            for update in app.stream(inputs, stream_mode="updates"):

                if isinstance(update, dict) and len(update) == 1:
                    node_name = list(update.keys())[0]
                    node_output = list(update.values())[0]

                    if isinstance(node_output, dict):
                        final_state.update(node_output)

                    if node_name != current_node:
                        current_node = node_name
                        status_area.info(f"⚙️ Executing: **{node_name}**")

            status_area.success("✅ Blog generation completed")

        final_md = final_state.get("final", "")

        if final_md:

           # Auto-select most recent file saved by backend
            latest_file = max(FILES_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime)
            st.session_state.selected_blog = latest_file

            # Stream display
            with blog_container:
                stream_placeholder = st.empty()
                streamed_text = ""

                for i in range(0, len(final_md), 60):
                    streamed_text = final_md[:i+60]
                    stream_placeholder.markdown(streamed_text)
                    time.sleep(0.01)

                st.download_button(
                    "⬇️ Download",
                    data=final_md.encode("utf-8"),
                    file_name=file_path.name,
                    mime="text/markdown",
                    use_container_width=True
                )

        else:
            with blog_container:
                st.warning("No output generated.")

    elif generate:
        status_area.warning("Please enter a topic.")