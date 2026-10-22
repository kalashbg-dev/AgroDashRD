import streamlit as st

from agrodash.web.state import require

require("admin")
st.title("admin")
st.info("En construcción.")
