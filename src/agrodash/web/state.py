"""Database access and the signed-in user for the Streamlit pages.

The user lives only in Streamlit's server-side session state: nothing is stored in the
browser, so reloading the page signs out. Every page calls require() itself, on top of
the navigation hiding pages a role cannot open.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from agrodash.accounts.models import Role
from agrodash.accounts.service import AuthenticationError, authenticate
from agrodash.infra.db import create_db_engine
from agrodash.infra.settings import get_settings
from agrodash.web.access import can_open

SIGN_IN_FAILED = "Correo o contraseña incorrectos."


@st.cache_resource
def _engine(url: str) -> Engine:
    return create_db_engine(url)


@contextmanager
def database() -> Iterator[Session]:
    with Session(_engine(get_settings().database_url)) as session:
        yield session


def current_user() -> dict | None:
    return st.session_state.get("user")


def current_role() -> Role | None:
    user = current_user()
    return Role(user["role"]) if user else None


def sign_in(email: str, password: str) -> bool:
    with database() as session:
        try:
            user = authenticate(session, email, password)
        except AuthenticationError:
            return False
        st.session_state["user"] = {
            "id": user.id,
            "email": user.email,
            "name": user.full_name,
            "role": str(user.role),
        }
    return True


def sign_out() -> None:
    st.session_state.pop("user", None)


def require(page: str) -> None:
    """Stop rendering unless the current user may open `page`."""
    if not can_open(current_role(), page):
        st.error("No tienes permiso para ver esta página.")
        st.stop()
