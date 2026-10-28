"""Catalog changes an admin makes: confirm aliases, rename, merge and retire products."""

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, aliased, selectinload

from agrodash.accounts.models import User
from agrodash.accounts.service import can_manage
from agrodash.catalog.approved_merges import APPROVED_MERGES
from agrodash.catalog.models import Market, Product, ProductAlias
from agrodash.ingestion.normalize import clean_name, name_key
from agrodash.prices.models import Price, PriceSource


class CatalogError(ValueError):
    """A catalog change that cannot be made, safe to show to the admin."""


def confirm_alias(session: Session, actor: User, alias_id: int) -> ProductAlias:
    _check(actor)
    alias = _get(session, ProductAlias, alias_id)
    alias.reviewed = True
    session.commit()
    return alias


def rename_product(session: Session, actor: User, product_id: int, name: str) -> Product:
    _check(actor)
    product = _get(session, Product, product_id)
    name = clean_name(name)[:120]
    if not name:
        raise CatalogError("The name cannot be empty")
    taken = session.scalar(select(Product.id).where(Product.name == name, Product.id != product.id))
    if taken:
        raise CatalogError(f"A product named {name!r} already exists")
    product.name = name
    session.commit()
    return product


def merge_products(session: Session, actor: User, source_id: int, target_id: int) -> Product:
    """Move every price and alias of `source` to `target` and retire `source`.

    For products the reports renamed ("Cilantro/ grande/ primera" became "Cilantro,
    primera"). The source stays, inactive, so old forecast runs keep their references.
    Its latest forecasts stay on the source until the next forecast run replaces them.
    It refuses when both products have an official price for the same market, week and
    type: choosing which one to drop is a decision, not a merge.
    """
    _check(actor)
    if source_id == target_id:
        raise CatalogError("A product cannot be merged into itself")
    return _merge(session, _get(session, Product, source_id), _get(session, Product, target_id))


def apply_approved_merges(session: Session) -> list[tuple[str, str]]:
    """Merge the approved product pairs (audit G7) wherever both are still apart.

    A database loaded before the approval, or any copy of it, catches up on its next
    load; pairs already merged are skipped. Returns (name, reason) for refused pairs.
    """
    aliases = session.scalars(select(ProductAlias).options(selectinload(ProductAlias.product)))
    owners = {alias.raw_name: alias.product for alias in aliases}
    problems = []
    for old, canon in APPROVED_MERGES:
        source, target = owners.get(name_key(old)), owners.get(name_key(canon))
        if (
            source is not None
            and target in (None, source)
            and source.name == old
            and not session.scalar(select(Product.id).where(Product.name == canon))
        ):
            # Already one product, loaded under the old name: show the approved one.
            source.name = canon
            session.commit()
        if source is None or target is None or source.id == target.id:
            continue
        try:
            _merge(session, source, target)
        except CatalogError as error:
            session.rollback()
            problems.append((old, str(error)))
            continue
        owners = {key: target if owner is source else owner for key, owner in owners.items()}
    return problems


def _merge(session: Session, source: Product, target: Product) -> Product:
    if not target.is_active:
        raise CatalogError("Cannot merge into a retired product")
    theirs = aliased(Price)
    clashes = session.scalar(
        select(func.count())
        .select_from(Price)
        .join(
            theirs,
            (theirs.market_id == Price.market_id)
            & (theirs.observed_on == Price.observed_on)
            & (theirs.price_type == Price.price_type),
        )
        .where(
            Price.product_id == source.id,
            Price.source == PriceSource.OFFICIAL,
            theirs.product_id == target.id,
            theirs.source == PriceSource.OFFICIAL,
        )
    )
    if clashes:
        raise CatalogError(
            f"{clashes} week{'s' if clashes > 1 else ''} where both products have an official "
            "price; fix or delete one of them first"
        )
    session.execute(update(Price).where(Price.product_id == source.id).values(product_id=target.id))
    session.execute(
        update(ProductAlias)
        .where(ProductAlias.product_id == source.id)
        .values(product_id=target.id, reviewed=True)
    )
    source.is_active = False
    session.commit()
    return target


def merge_family(
    session: Session, actor: User, target_id: int, member_ids: list[int]
) -> list[tuple[str, str]]:
    """Merge every member of a product family into `target_id`.

    Each member is merged on its own, so one that clashes does not stop the rest;
    returns (name, reason) for the members left unmerged.
    """
    _check(actor)
    problems = []
    for member_id in member_ids:
        if member_id == target_id:
            continue
        try:
            merge_products(session, actor, member_id, target_id)
        except CatalogError as error:
            session.rollback()
            problems.append((_get(session, Product, member_id).name, str(error)))
    return problems


def set_product_active(session: Session, actor: User, product_id: int, *, active: bool) -> Product:
    _check(actor)
    product = _get(session, Product, product_id)
    product.is_active = active
    session.commit()
    return product


def set_market_active(session: Session, actor: User, market_id: int, *, active: bool) -> Market:
    _check(actor)
    market = _get(session, Market, market_id)
    market.is_active = active
    session.commit()
    return market


def _check(actor: User) -> None:
    if not (actor.is_active and can_manage(actor.role)):
        raise CatalogError("Your account cannot change the catalog")


def _get(session: Session, model, row_id: int):
    row = session.get(model, row_id)
    if row is None:
        raise CatalogError(f"Unknown {model.__tablename__[:-1]}")
    return row
