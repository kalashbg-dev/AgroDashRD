"""Products the reports listed under different names over the years."""

from dataclasses import dataclass
from datetime import date
from itertools import combinations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agrodash.catalog.models import Product
from agrodash.ingestion.normalize import name_key, same_product_name, word_set_key
from agrodash.prices.models import Price

# Early reports left out the grade when only "primera" was sold.
DEFAULT_GRADES = {"primera", "primero"}


@dataclass(frozen=True)
class FamilyMember:
    id: int
    name: str
    first: date
    last: date
    prices: int


@dataclass(frozen=True)
class Family:
    """Names of one product; the first member is the one the others merge into."""

    members: tuple[FamilyMember, ...]
    # Safe: every name has the same words. Otherwise a manager should look first.
    safe: bool

    @property
    def target(self) -> FamilyMember:
        return self.members[0]


def product_families(session: Session) -> list[Family]:
    """Active products that look like one product under several names, safe ones first."""
    rows = session.execute(
        select(
            Product.id,
            Product.name,
            Product.base_unit,
            func.min(Price.observed_on),
            func.max(Price.observed_on),
            func.count(),
        )
        .join(Price, Price.product_id == Product.id)
        .where(Product.is_active)
        .group_by(Product.id, Product.name, Product.base_unit)
    ).all()
    members = [FamilyMember(row[0], row[1], row[3], row[4], row[5]) for row in rows]
    words = [word_set_key(name_key(member.name)) for member in members]
    parent = list(range(len(members)))
    unsafe = set()

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, j in combinations(range(len(members)), 2):
        if rows[i][2] != rows[j][2]:
            continue
        if words[i] == words[j]:
            safe = True
        elif _suggested(words[i], words[j]):
            safe = False
        else:
            continue
        a, b = root(i), root(j)
        joined_unsafe = not safe or a in unsafe or b in unsafe
        parent[b] = a
        unsafe.discard(b)
        if joined_unsafe:
            unsafe.add(a)

    groups: dict[int, list[FamilyMember]] = {}
    for index, member in enumerate(members):
        groups.setdefault(root(index), []).append(member)
    families = []
    for group_root, group in groups.items():
        if len(group) < 2:
            continue
        # The name in use most recently is the one the reports use today.
        target = max(group, key=lambda m: (m.last, m.prices))
        others = sorted((m for m in group if m is not target), key=lambda m: (m.first, m.name))
        families.append(Family((target, *others), safe=group_root not in unsafe))
    return sorted(families, key=lambda f: (not f.safe, f.target.name))


def _suggested(left: str, right: str) -> bool:
    """Names that differ only by a missing "primera" or by a typo in one word."""
    left_words = [w for w in left.split() if w not in DEFAULT_GRADES]
    right_words = [w for w in right.split() if w not in DEFAULT_GRADES]
    return left_words == right_words or same_product_name(left, right)
