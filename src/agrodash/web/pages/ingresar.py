import streamlit as st

from agrodash.web.state import SIGN_IN_FAILED, current_user, sign_in, sign_out


def _attempt_sign_in() -> None:
    succeeded = sign_in(st.session_state.get("email", ""), st.session_state.get("password", ""))
    # Never keep the password in the session once it was used.
    st.session_state["password"] = ""
    st.session_state["sign_in_failed"] = not succeeded


user = current_user()
if user:
    st.title("Sesión iniciada")
    st.write(f"Has ingresado como **{user['name'] or user['email']}** ({user['role'].lower()}).")
    st.button("Cerrar sesión", key="sign_out", on_click=sign_out)
else:
    st.title("Ingresar")
    st.caption("Solo técnicos y administradores necesitan cuenta; los precios son públicos.")
    st.text_input("Correo", key="email", autocomplete="email")
    st.text_input("Contraseña", key="password", type="password", autocomplete="current-password")
    st.button("Ingresar", key="sign_in", type="primary", on_click=_attempt_sign_in)
    if st.session_state.pop("sign_in_failed", False):
        st.error(SIGN_IN_FAILED)
