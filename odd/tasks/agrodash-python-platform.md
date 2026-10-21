# AgroDashRD Python platform

Objective: rebuild AgroDashRD as a single Python app (Streamlit, mobile-first web) with a clean data pipeline and validated forecasting.
Branch: one stacked branch per phase (`feat/f0-foundations` → `feat/f1-schema` → ...); user pushes and opens PRs from VS Code. Delivery: stacked-to-main. Test runner: `uv run pytest`.

## Specs

- S1 (F0, audit points 1 and 7): "este proyecto quiero hacerlo en python y llevarlo a una version mobile tipo webapp si aplica, o usa streamlit si sigue siendo buena opcion, no lo compliques". Fix `.gitignore` (`lib/` hid Flutter sources), remove `mobile_app/` (Flutter, no code), unify dependencies into one `pyproject.toml` managed by `uv` on Python 3.12, `src/agrodash` package, settings that fail fast in production, pytest + ruff, GitHub Actions CI.
- S2 (F1, point 3): relational schema with `Numeric` prices, `Date` weeks, units with conversion factors, product aliases, price source and status, uniqueness on (product, market, date, price_type, source); Alembic migrations; SQLite for development, PostgreSQL for production. Port useful models from `backend/`.
- S3 (F2, point 2): accounts with roles (AGRICULTOR, TECNICO, ADMIN, SUPERADMIN), argon2 hashing, lockout after repeated failed logins, `create-admin` CLI, docker-compose without exposed DB or default passwords. Public read without login; login only for reporting, uploading and validating. Remove `backend/` once ported. Close with the `security-review` skill.
- S4 (F3, point 4): hybrid ingestion. Parsers for xlsx/xls/csv (Latin-1, `;`, headerless wholesale CSV, wide retail sheet), normalization of week words with typos, rubros, units and "N/D"; product alias catalog drafted by fuzzy matching and marked unreviewed; quarantine table with reasons; idempotent runs keyed by file hash; public Drive download by file id; admin upload; field reports stored as pending with outlier check against the official series; full history load. User request: "Que haremos con esta conclusion: el problema principal hoy no es qué modelo usar sino la calidad de los datos. Hace falta un catálogo de productos con sus variantes de nombre y una tabla de conversión de unidades. Sin eso, cualquier modelo aprende ruido." F3 closes with a data-quality report (rejections by reason, unmapped names, weekly coverage per series).
- S5 (F4, point 5): correct weekly dates; seasonal-naive baseline; AutoETS/AutoARIMA/AutoTheta (statsforecast); rolling-origin backtest with MASE; per-series model selection; global LightGBM (mlforecast); predictions table with intervals and model version; batch job. Only series that pass the data-quality threshold are trained; others are shown as "sin pronóstico".
- S6 (F5): Streamlit mobile-first pages: Precios, Pronóstico, Reportar precio, Admin (upload, validate reports, rejects, users). Port useful legacy charts (use `dataviz` skill), verify at narrow viewport, delete `legacy/`.
- S7 (F6, points 6 and 7): single Docker image for app and job, docs (README, deploy guide, audit report) updated, final `code-review`.

Assumption: the wholesale CSV belongs to "Mercado Nuevo" (legacy assumption) until the user confirms.

## Tasks

| ID | Specs | Route | Status | Commit |
|----|-------|-------|--------|--------|
| T0 | S1 | inline | done | a1c8010, 4cd34b6, 3632910, follow-up fix |
| T1 | S2 | inline | pending | |
| T2 | S3 | inline | pending | |
| T3 | S4 | inline | pending | |
| T4 | S5 | inline | pending | |
| T5 | S6 | inline | pending | |
| T6 | S7 | inline | pending | |

Route evidence: phases depend on each other (no independent parallel units), so no writer trigger fires; work stays inline.

## Log

- L1 (2026-10-21, user): "OK. me gusta tu hallazgo. este proyecto quiero hacerlo en python y llevarlo a una version mobile tipo webapp si aplica, o usa streamlit si sigue siendo buena opcion, no lo compliques. Tienes 7 puntos tocados en este informe auditoria. EMPIEZA A CUMPLIRLOS UNO POR UNO, Resuelve todo lo que sea estructural y d arquitectura que no necesite mi intervencion. Acepto tus sugerencias todas. si tienes skills de ciberseguridad, de diseño, y de lo que dices que falta, usalas. Reformula el plan y te doy el visto bueno. Piensa profundamente"
- L2 (2026-10-21, user): "me gusta este F0 a F6, se queda algo de lado. Que haremos con esta conclusion: el problema principal hoy no es qué modelo usar sino la calidad de los datos. Hace falta un catálogo de productos con sus variantes de nombre y una tabla de conversión de unidades. Sin eso, cualquier modelo aprende ruido.Es un enunciado que menciona algo que se debe resolver. Dentro de F3 debe estar, si es asi, obvia lo que digo pues ya lo pensaste. PRocede"
- L4 (T0 evidence): settings tests RED (module missing) then GREEN, 4 passed; fresh clone `uv sync --locked`, `ruff check`, `ruff format --check`, `pytest` all pass; `git check-ignore` confirms `mobile_app/lib/`, `src/`, `tests/fixtures/*.csv` are no longer ignored. Assess: high (CI shell, removed executable script); native review lenses risk + reliability awaiting user consent.
- L5 (T0 review): F0 split into three commits because the first candidate exceeded the reviewer context budget. a1c8010 approved (risk lens, no findings). 4cd34b6 (generated lockfile and Flutter deletions) unavailable: `lens_context_budget_exceeded` even alone. 3632910 approved (risk + reliability) with warnings, fixed in a follow-up commit: production guard now requires a postgresql scheme (empty, upper-case SQLite, mysql rejected; 12 tests pass), data files ignored repo-wide except `tests/fixtures/**`, CI actions pinned to commit SHAs (checkout v5.1.0, setup-uv v6.8.0). `start_dev.sh` stays until `backend/` is removed in F2.
- L3: plan approved; data-quality gate added to S4/S5. Engram mirror pending: `mem_current_project` is ambiguous from `~/Downloads` (agrodashrd vs another repo).
