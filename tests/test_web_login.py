from pathlib import Path

import pytest
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from agrodash.accounts.models import Role
from agrodash.accounts.service import create_user

PAGES = Path(__file__).parent.parent / "src" / "agrodash" / "web" / "pages"
PASSWORD = "correct horse battery"


@pytest.fixture
def technician(cli_database):
    with Session(cli_database) as session:
        create_user(session, "tecnico@example.com", PASSWORD, Role.TECNICO, "Ana Pérez")


def login(email, password) -> AppTest:
    app = AppTest.from_file(str(PAGES / "ingresar.py")).run()
    app.text_input(key="email").input(email)
    app.text_input(key="password").input(password)
    app.button(key="sign_in").click()
    return app.run()


def test_signing_in_keeps_the_user_in_the_session(technician):
    app = login(" Tecnico@Example.com ", PASSWORD)

    assert not app.exception
    assert app.session_state["user"] == {
        "id": 1,
        "email": "tecnico@example.com",
        "name": "Ana Pérez",
        "role": "TECNICO",
    }


@pytest.mark.parametrize(
    ("email", "password"),
    [("tecnico@example.com", "wrong password!!"), ("nadie@example.com", PASSWORD)],
)
def test_a_failed_sign_in_shows_one_generic_message(technician, email, password):
    app = login(email, password)

    assert [error.value for error in app.error] == ["Correo o contraseña incorrectos."]
    assert "user" not in app.session_state


def test_signing_out_forgets_the_user(technician):
    app = login("tecnico@example.com", PASSWORD)

    app.button(key="sign_out").click()
    app.run()

    assert "user" not in app.session_state


def test_the_app_starts_for_an_anonymous_visitor(cli_database):
    app = AppTest.from_file(str(PAGES.parent / "app.py")).run()

    assert not app.exception


def test_web_command_starts_streamlit_on_the_app(monkeypatch):
    from agrodash import cli

    calls = []
    monkeypatch.setattr(cli.os, "execv", lambda path, args: calls.append(args))

    cli.main(["web", "--port", "8600"])

    (args,) = calls
    assert args[1:4] == ["-m", "streamlit", "run"]
    assert args[4].endswith("agrodash/web/app.py")
    assert args[5:] == ["--server.port", "8600", "--server.headless", "true"]
