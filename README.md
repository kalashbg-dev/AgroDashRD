# AgroDashRD - Dominican Agricultural Sector Analysis Dashboard 🌱

![Agrodash](Screenshots/Agrodash-main.png)

## What is AgroDashRD?

AgroDashRD is a comprehensive platform for visualizing and analyzing agricultural sector data in the Dominican Republic. It is being rebuilt as a single Python application: a mobile-first **Streamlit** web app on top of a clean data pipeline and validated price forecasts.

## Architecture

One Python package, `src/agrodash`, organized by feature. Business logic does not depend on the UI, so the Streamlit layer can be replaced later without touching it.

```
src/agrodash/
  catalog/      products, name aliases, units, markets
  prices/       prices, field reports and validation
  ingestion/    Google Drive and upload sources, parsers, normalizers, quarantine
  forecasting/  backtesting, models, batch job, stored predictions
  accounts/     users, roles, password hashing, login lockout
  infra/        settings, database, migrations
  web/          Streamlit pages (presentation only)
```

The migration runs in phases (F0 to F6); progress is tracked in [odd/tasks/agrodash-python-platform.md](odd/tasks/agrodash-python-platform.md). The FastAPI backend and the empty Flutter app were removed; `legacy/` (original Streamlit app) stays for reference until the new web app reaches parity.

## Features

### For Farmers 👨‍🌾
- **Current Prices**: Check real-time market prices.
- **Harvest Calculator**: Estimate harvest value.
- **Best Markets**: Find the most profitable markets.

### For Professionals 📊
- **Value Chain Analysis**: Track product flow.
- **Price Forecasts**: ML-powered price predictions.
- **Statistical Analysis**: In-depth metrics.

## Getting Started

### Prerequisites
- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 automatically)
- Docker & Docker Compose (optional): `docker compose up -d db` starts a local PostgreSQL after you set `POSTGRES_PASSWORD` in `.env`

### Development

```bash
uv sync                 # create .venv and install dependencies
cp .env.example .env    # adjust settings if needed
uv run alembic upgrade head   # create or upgrade the database schema
uv run agrodash create-admin --email you@example.com --superadmin   # first account
uv run pytest           # run the test suite
uv run ruff check .     # lint
uv run ruff format .    # format
```

### Configuration

Settings are read from `AGRODASH_*` environment variables or `.env` (see [.env.example](.env.example)):

| Variable | Default | Notes |
|---|---|---|
| `AGRODASH_ENV` | `development` | `development` or `production` |
| `AGRODASH_DATABASE_URL` | `sqlite:///agrodash.db` | Production refuses to start on SQLite |

Reading prices and forecasts needs no account. Accounts exist for people who report prices (TECNICO), manage data (ADMIN) or own the platform (SUPERADMIN). Passwords are hashed with Argon2id, and five failed logins lock an account for 15 minutes.

Schema changes go through Alembic migrations in `src/agrodash/infra/migrations`: after editing a model, run `uv run alembic revision --autogenerate -m "<change>"` and review the generated file. Tests run every schema test on SQLite, and also on PostgreSQL when `AGRODASH_TEST_POSTGRES_URL` is set (CI does this).

## Documentation

- **[INFORME_AUDITORIA.md](INFORME_AUDITORIA.md)**: Audit report.
- Deployment and configuration guides are rewritten in phase F6; the current ones still describe the FastAPI + Flutter design.

## Support

For questions, please open an issue or contact the development team.

---
Developed with ❤️ for the Dominican agricultural sector.
