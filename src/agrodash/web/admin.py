"""Actions behind the administration page, kept out of the page so they can be tested."""

from sqlalchemy.orm import Session

from agrodash.accounts.models import User
from agrodash.ingestion.models import IngestionTrigger
from agrodash.ingestion.service import IngestionError, ingest_file, loaded_run

REASONS = {
    "duplicate_price": "Precio repetido en la misma semana",
    "invalid_date": "Fecha ilegible",
    "invalid_price": "Precio ilegible",
    "unexpected_layout": "Fila con formato inesperado",
    "unit_mismatch": "Unidad distinta a la de su serie",
    "unknown_unit": "Unidad sin tamaño conocido",
}


def load_report(
    session: Session, file_name: str, content: bytes, trigger: IngestionTrigger, actor_id: int
) -> tuple[str, str]:
    """Load a report file and describe the result as (kind, message) for st.<kind>."""
    if loaded_run(session, content):
        return "info", f"{file_name} ya estaba cargado."
    try:
        run = ingest_file(session, content, file_name, trigger, session.get(User, actor_id))
    except IngestionError as error:
        return "error", f"{file_name}: no se pudo cargar ({error})."
    return (
        "success",
        f"{file_name}: {run.prices_loaded} precios cargados, {run.rows_rejected} filas en "
        f"cuarentena, {run.values_missing} valores vacíos.",
    )
