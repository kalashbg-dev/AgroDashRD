import streamlit as st

from agrodash.web.state import require

require("catalogo")
st.title("catalogo")
st.info("En construcción.")
