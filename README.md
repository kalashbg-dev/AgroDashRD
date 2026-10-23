# AgroDashRD

Agricultural prices of the Dominican Republic, readable on a phone: the latest official wholesale and retail prices, how they move through the year, where each product is cheapest, and what to expect in the coming weeks and months. Anyone can read it; technicians report field prices and admins keep the data clean.

| Precios | Temporadas | Mercados | Pronóstico |
|---|---|---|---|
| ![Precios](Screenshots/precios.png) | ![Temporadas](Screenshots/temporadas.png) | ![Mercados](Screenshots/mercados.png) | ![Pronóstico](Screenshots/pronostico.png) |

## Quick start (Docker)

Needs Docker Engine 20.10 or later with Compose v2. The first `up` builds the image (1.3 GB, a few minutes).

```bash
echo "POSTGRES_PASSWORD=$(openssl rand -hex 16)" > .env
docker compose up -d                       # PostgreSQL, schema migration, web app on http://localhost:8501
docker compose run --rm web create-admin --email you@example.com --superadmin
docker compose run --rm weekly             # migrate, load the official reports from Google Drive, forecast (about 10 minutes)
```

Production setup (HTTPS, the weekly schedule, backups and updates) is in **[GUIA_DESPLIEGUE.md](GUIA_DESPLIEGUE.md)**. How to use each page is in **[GUIA_USO.md](GUIA_USO.md)**.

## Pages

| Page | Who | What it answers |
|---|---|---|
| Precios | everyone | Latest price per market, history over a chosen period, comparison with other products, biggest four-week rises and falls, CSV downloads |
| Pronóstico | everyone | Price in 4 weeks, 3 months and 12 months, with 80% and 95% ranges and whether that month is usually dear or cheap |
| Temporadas | everyone | Months of high and low prices, year by year, and for the whole rubro |
| Mercados | everyone | Cheapest and dearest markets, wholesale vs retail, what a harvest is worth |
| Reportar precio | TECNICO and up | Field price reports, pending until an admin validates them |
| Administración | ADMIN, SUPERADMIN | Load report files, validate field reports, quarantine, accounts |
| Catálogo | ADMIN, SUPERADMIN | Merge products listed under different names, confirm name aliases, rename, retire |

## How the data is handled

- **Sources.** The ministry's wholesale and retail reports (`.csv`, `.xlsx`, `.xls`), from Google Drive (`agrodash ingest --drive`), files, or an upload in Administración. Loading the same file twice does nothing.
- **Cleaning.** Spelling variants of products, rubros, units and week words are normalized; prices are converted to a base unit (pound, unit or liter). Clear export mistakes are repaired (unreadable dates between agreeing rows, "ñ" exported as ";", Excel error cells). Rows that cannot be trusted go to quarantine with a reason instead of being loaded.
- **Products across report eras.** Reports renamed products over the years ("Ají/ Cachucha/ verde" became "Ají (Cachucha)/ verde/ primera"). Names with the same words are matched automatically; near matches are suggested in Catálogo → Repetidos, where an admin merges them so each product keeps one history. Sizes and grades are never merged.
- **Weeks.** Reports publish weeks 1 to 4 of each month and sometimes a fifth. Forecasts count report periods, not calendar days.
- **Quality.** `agrodash quality-report` summarizes rejections, unreviewed names, likely renames and which series are ready to forecast (two years of prices, 75% of weeks covered, a price in the last 8 weeks).

## Forecasts

The forecast is the last published price, with ranges from how the series moved before. Backtests on the official history decided this:

| Horizon | Better than the last price? | 80% range held | 95% range held |
|---|---|---|---|
| 4 weeks | No model (AutoETS, Theta, gradient boosting) | about 82% | about 93% |
| 3 months | No, not even adjusted by season | 84% | 96% |
| 12 months | No | 82% | 92% |

Weeks 1-4 use the last year of moves; months 2-12 use the whole history and need three years of prices. Every run re-measures the other models against the last price, so a change would show. Series that are not ready show "sin pronóstico" with the reason.

## Development

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12).

```bash
uv sync                                   # create .venv with every dependency
cp .env.example .env                      # SQLite by default
uv run agrodash migrate                   # create or update the schema
uv run agrodash create-admin --email you@example.com --superadmin
uv run agrodash ingest --drive            # or: uv run agrodash ingest mayorista.xlsx minorista.xls
uv run agrodash forecast                  # --skip-benchmark takes seconds instead of minutes
uv run agrodash web                       # http://localhost:8501
uv run pytest                             # tests
uv run ruff check . && uv run ruff format .
```

For PostgreSQL locally, `docker compose up -d db` and point `AGRODASH_DATABASE_URL` at it. Tests also run every database test on PostgreSQL when `AGRODASH_TEST_POSTGRES_URL` names a database ending in `_test` (CI does this).

Schema changes go through Alembic: edit a model, run `uv run alembic revision --autogenerate -m "<change>"`, review the generated file in `src/agrodash/infra/migrations`.

### Configuration

| Variable | Default | Notes |
|---|---|---|
| `AGRODASH_ENV` | `development` | `production` refuses to start on SQLite |
| `AGRODASH_DATABASE_URL` | `sqlite:///agrodash.db` | `postgresql+psycopg://user:password@host:5432/agrodash` |
| `AGRODASH_DRIVE_FILE_IDS` | the official wholesale and retail reports | Comma-separated ids of Drive files shared with "Anyone with the link" |

### Layout

```
src/agrodash/
  catalog/      products, families of names, aliases, units, markets
  prices/       prices, field reports, seasons, market comparisons, page queries
  ingestion/    Drive and file sources, parsers, normalizers, repairs, quarantine, quality
  forecasting/  report-period series, forecasts and ranges, benchmark, batch job
  accounts/     users, roles, Argon2id hashing, login lockout
  infra/        settings, database, migrations
  web/          Streamlit pages, charts and theme (presentation only)
```

Business logic does not import Streamlit, so it is tested without it. The work is tracked in [odd/tasks/agrodash-python-platform.md](odd/tasks/agrodash-python-platform.md). The original Streamlit app, the FastAPI backend and the empty Flutter app were replaced; the old app stays readable in git history (`git show 75c584b:legacy/app.py`), and [INFORME_AUDITORIA.md](INFORME_AUDITORIA.md) records what was found and how it was resolved.

## Accounts and security

Reading needs no account. TECNICO accounts report prices, ADMIN accounts manage data and technicians, SUPERADMIN accounts also manage admins. Passwords are hashed with Argon2id; five failed logins lock an account for 15 minutes; every page re-reads the signed-in account, so a deactivated account loses access at once.
