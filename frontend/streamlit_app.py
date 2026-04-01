from __future__ import annotations

import json

import requests
import streamlit as st

st.set_page_config(page_title="AI Productivity Agent", page_icon="🧭", layout="wide")
st.title("AI Personal Productivity Agent")
st.caption("Agent + Tools + Skills + RAG + Memory demo")

backend_url = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000")
session_id = st.sidebar.text_input("Session ID", "demo-session")
user_id = st.sidebar.text_input("User ID", "cn-user")

message = st.text_area("Your request", "帮我规划一个周末洛杉矶一日游，考虑天气、人流，并生成行程表。我的预算是200。")

if st.button("Run Agent", use_container_width=True):
    payload = {"session_id": session_id, "user_id": user_id, "message": message}
    try:
        resp = requests.post(f"{backend_url}/chat", json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()

        st.subheader("Answer")
        st.code(data["answer"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan Steps")
            st.write(data["plan"])
            st.subheader("Used Tools")
            st.write(data["used_tools"])

        with col2:
            st.subheader("Retrieved Docs")
            st.write(data["retrieved_docs"])

        with st.expander("Debug Payload"):
            st.code(json.dumps(data["debug"], ensure_ascii=False, indent=2))
    except Exception as exc:
        st.error(f"Request failed: {exc}")
