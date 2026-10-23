from alembic import context
from sqlalchemy import create_engine

import agrodash.accounts.models  # noqa: F401  (registers tables on Base.metadata)
import agrodash.catalog.models  # noqa: F401
import agrodash.forecasting.models  # noqa: F401
import agrodash.ingestion.models  # noqa: F401
import agrodash.prices.models  # noqa: F401
from agrodash.infra.db import Base
from agrodash.infra.settings import get_settings

url = context.config.get_main_option("sqlalchemy.url") or get_settings().database_url
# SQLite cannot ALTER most constraints; batch mode rebuilds tables instead.
render_as_batch = url.startswith("sqlite")


def run_migrations_offline() -> None:
    context.configure(
        url=url,
        target_metadata=Base.metadata,
        literal_binds=True,
        render_as_batch=render_as_batch,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # A plain engine, without the app's SQLite foreign-key enforcement: batch mode
    # rebuilds tables (drop and recreate), which enforcement blocks as soon as other
    # tables reference rows. Rows keep their ids, so references stay valid.
    engine = create_engine(url)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=Base.metadata,
            render_as_batch=render_as_batch,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
