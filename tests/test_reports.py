from datetime import date, timedelta
from decimal import Decimal

import pytest

from agrodash.accounts.models import Role
from agrodash.accounts.service import create_user
from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, Unit
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType
from agrodash.prices.reports import ReportError, pending_reports, report_price, review_report

PASSWORD = "correct horse battery"
WEEK = date(2026, 9, 1)


@pytest.fixture
def catalog(session):
    rice = Product(name="Arroz Selecto", category="Cereales", base_unit=BaseUnit.POUND)
    market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
    pound = Unit(code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1))
    sack = Unit(code="100-lb", label="100 lb", base_unit=BaseUnit.POUND, factor=Decimal(100))
    piece = Unit(code="1-unit", label="1 unit", base_unit=BaseUnit.UNIT, factor=Decimal(1))
    session.add_all([rice, market, pound, sack, piece])
    for weeks_ago, amount in enumerate(["30", "31", "29", "30"]):
        session.add(
            Price(
                product=rice,
                market=market,
                unit=pound,
                observed_on=WEEK - timedelta(weeks=weeks_ago),
                price_type=PriceType.RETAIL,
                source=PriceSource.OFFICIAL,
                status=PriceStatus.VALIDATED,
                amount=Decimal(amount),
                price_per_base_unit=Decimal(amount),
            )
        )
    session.commit()
    return {"product": rice, "market": market, "pound": pound, "sack": sack, "piece": piece}


@pytest.fixture
def users(session):
    return {
        role: create_user(session, f"{role.lower()}@example.com", PASSWORD, role) for role in Role
    }


def report(session, catalog, reporter, amount="31", unit="pound", observed_on=WEEK, **extra):
    return report_price(
        session,
        reporter,
        product_id=catalog["product"].id,
        market_id=catalog["market"].id,
        unit_id=catalog[unit].id,
        amount=Decimal(amount),
        observed_on=observed_on,
        price_type=PriceType.RETAIL,
        **extra,
    )


def test_a_field_report_is_stored_pending_with_its_reporter(session, catalog, users):
    price = report(session, catalog, users[Role.TECNICO], amount="3150", unit="sack")

    stored = session.get(Price, price.id)
    assert (stored.source, stored.status, stored.reporter_id) == (
        PriceSource.FIELD_REPORT,
        PriceStatus.PENDING,
        users[Role.TECNICO].id,
    )
    assert (stored.amount, stored.price_per_base_unit, stored.is_outlier) == (
        Decimal("3150.00"),
        Decimal("31.5000"),
        False,
    )


@pytest.mark.parametrize(("amount", "outlier"), [("44", False), ("46", True), ("14", True)])
def test_reports_far_from_the_official_median_are_flagged(session, catalog, users, amount, outlier):
    # The official median of the last weeks is 30; more than 50% away is an outlier.
    price = report(session, catalog, users[Role.TECNICO], amount=amount)

    assert price.is_outlier is outlier


def test_reports_without_an_official_series_are_not_flagged(session, catalog, users):
    long_ago = WEEK - timedelta(weeks=20)

    assert (
        report(session, catalog, users[Role.TECNICO], "90", observed_on=long_ago).is_outlier
        is False
    )


@pytest.mark.parametrize("role", [Role.TECNICO, Role.ADMIN, Role.SUPERADMIN])
def test_reporting_roles_can_report(session, catalog, users, role):
    assert report(session, catalog, users[role]).status is PriceStatus.PENDING


def test_farmers_and_inactive_accounts_cannot_report(session, catalog, users):
    users[Role.TECNICO].is_active = False

    for role in (Role.AGRICULTOR, Role.TECNICO):
        with pytest.raises(ReportError, match="cannot report prices"):
            report(session, catalog, users[role])
    assert session.query(Price).filter_by(source=PriceSource.FIELD_REPORT).count() == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"amount": "0"}, "Price must be greater than zero"),
        ({"amount": "-3"}, "Price must be greater than zero"),
        ({"observed_on": date.today() + timedelta(days=1)}, "cannot be in the future"),
        ({"unit": "piece"}, "Arroz Selecto retail prices are per lb; choose a unit in lb"),
    ],
)
def test_invalid_reports_are_refused(session, catalog, users, changes, message):
    with pytest.raises(ReportError, match=message):
        report(session, catalog, users[Role.TECNICO], **changes)


def test_unknown_catalog_ids_are_refused(session, catalog, users):
    with pytest.raises(ReportError, match="Unknown product, market or unit"):
        report_price(
            session,
            users[Role.TECNICO],
            product_id=999,
            market_id=catalog["market"].id,
            unit_id=catalog["pound"].id,
            amount=Decimal("30"),
            observed_on=WEEK,
            price_type=PriceType.RETAIL,
        )


@pytest.mark.parametrize(
    ("approve", "status"), [(True, PriceStatus.VALIDATED), (False, PriceStatus.REJECTED)]
)
def test_admins_validate_or_reject_pending_reports(session, catalog, users, approve, status):
    price = report(session, catalog, users[Role.TECNICO])

    review_report(session, users[Role.ADMIN], price.id, approve=approve)

    assert session.get(Price, price.id).status is status
    assert pending_reports(session) == []


def test_only_managers_review_and_only_pending_field_reports(session, catalog, users):
    price = report(session, catalog, users[Role.TECNICO])
    official = session.query(Price).filter_by(source=PriceSource.OFFICIAL).first()

    with pytest.raises(ReportError, match="cannot review reports"):
        review_report(session, users[Role.TECNICO], price.id, approve=True)
    with pytest.raises(ReportError, match="not a pending field report"):
        review_report(session, users[Role.ADMIN], official.id, approve=False)
    review_report(session, users[Role.ADMIN], price.id, approve=True)
    with pytest.raises(ReportError, match="not a pending field report"):
        review_report(session, users[Role.ADMIN], price.id, approve=False)
    assert session.get(Price, official.id).status is PriceStatus.VALIDATED


def test_pending_reports_list_outliers_first(session, catalog, users):
    normal = report(session, catalog, users[Role.TECNICO], "31")
    outlier = report(session, catalog, users[Role.TECNICO], "90")

    assert pending_reports(session) == [outlier, normal]
